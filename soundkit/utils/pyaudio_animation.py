import os
import wave
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import pyaudio
import tkinter as tk
from tkinter import filedialog

plt.style.use('bmh')
LINE_MINMAX = [-1.1, 1.1]

def simple_trigger_processor_vad(frame_data, threshold=0.05):
    energy = np.sqrt(np.mean(frame_data[:, 0] ** 2))
    vad_val = min(energy / threshold, 1.0) * 0.8
    output = np.zeros((frame_data.shape[0], 2))
    output[:, 1] = vad_val
    return output

class AudioShowClass:
    def __init__(
            self,
            record_seconds=6,
            sample_rate=16000,
            frame_size=160,
            wave_output_filename="",
            non_stop=False,
            proc_st=None,
            reset_st=None,
            title="VAD"):

        self.record_seconds = record_seconds
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.wave_output_filename = wave_output_filename
        self.frame_time_record = float(frame_size) / float(sample_rate)
        self.frame_time_replay = 0.05
        self.non_stop = non_stop
        self.proc_st = proc_st
        self.reset_st = reset_st

        self.num_blks = int(self.sample_rate / self.frame_size * self.record_seconds)
        self.data_buffer = np.zeros((self.sample_rate * self.record_seconds, 2), dtype=float)
        self.const_data_buffer = np.arange(self.sample_rate * self.record_seconds) / self.sample_rate
        self.counts_frames = 0
        self.lock_button = 0
        self.start_record = 0
        self.draw_lock = 0
        self.replay_channel = 0  # 0 or 1

        self.fig, (ax_wave, ax_vad) = plt.subplots(nrows=2, figsize=(10, 6), sharex=True)
        self.fig.tight_layout(rect=[0, 0.25, 1, 0.95])

        for ax in [ax_wave, ax_vad]:
            ax.set_xlim((0, self.record_seconds))
            ax.set_ylim(LINE_MINMAX)
            ax.axhline(y=-1.0, color='black', linewidth=0.5)
            ax.axhline(y=1.0, color='black', linewidth=0.5)

        ax_wave.set_title("Waveform (Channel 0)")
        ax_wave.set_ylabel("Amplitude")
        self.line_data, = ax_wave.plot([], [], lw=0.2, color='blue')
        self.line_stop, = ax_wave.plot([], [], lw=0.2, color='k')

        ax_vad.set_title(f"{title} Signal (Channel 1)")
        ax_vad.set_ylabel("Processed")
        ax_vad.set_xlabel("Time (Seconds)")
        self.line_trig, = ax_vad.plot([], [], lw=0.5, color='red')
        self.line_stop_vad, = ax_vad.plot([], [], lw=0.2, color='k')

        self.fig.canvas.mpl_connect('close_event', self.handle_close)

        def make_button(pos, name, callback_func):
            ax_button = plt.axes(pos)
            button = Button(ax_button, name, color='w', hovercolor='aliceblue')
            button.label.set_fontsize(16)
            button.on_clicked(callback_func)
            return button

        num_buttons = 5
        gap_ratio = 0.02
        button_width_ratio = (1.0 - (num_buttons + 1) * gap_ratio) / num_buttons
        button_height = 0.07
        button_y = 0.05

        positions = []
        for i in range(num_buttons):
            x = gap_ratio + i * (button_width_ratio + gap_ratio)
            positions.append([x, button_y, button_width_ratio, button_height])

        self.button_saveas = make_button(positions[0], 'save as', self.callback_saveas)
        self.button_record = make_button(positions[1], 'record', self.callback_record)
        self.button_stop   = make_button(positions[2], 'stop', self.callback_stop)
        self.button_replay = make_button(positions[3], 'replay', self.callback_replay)
        self.button_toggle_channel = make_button(positions[4], 'channel: 0', self.callback_toggle_channel)

        plt.show()

    def handle_close(self, event):
        if self.start_record == 1:
            self.stop_recording()
        print('Window closed')

    def stop_recording(self):
        if self.reset_st is not None:
            self.reset_st()
        self.start_record = 0

    def callback_stop(self, event):
        self.stop_recording()

        if hasattr(event, "inaxes") and event.inaxes is not None:
            event.inaxes.figure.canvas.draw_idle()

    def callback_saveas(self, event):
        default_dir = os.getcwd()
        default_filename = "speech.wav"

        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            initialdir=default_dir,
            initialfile=default_filename,
            filetypes=[("WAV files", "*.wav")],
            title="Save WAV file as"
        )
        root.update()
        root.destroy()

        if file_path:
            self.wave_output_filename = file_path
            print(f"Output path set to: {self.wave_output_filename}")
        else:
            print("No file selected. Will use default: ./speech.wav")

        if hasattr(event, "inaxes") and event.inaxes is not None:
            event.inaxes.figure.canvas.draw_idle()

    def callback_toggle_channel(self, event):
        self.replay_channel = 1 - self.replay_channel
        self.button_toggle_channel.label.set_text(f'channel: {self.replay_channel}')
        print(f"Replay channel set to: {self.replay_channel}")
        if hasattr(event, "inaxes") and event.inaxes is not None:
            event.inaxes.figure.canvas.draw_idle()

    def callback_record(self, event):
        """Start recording audio and processing it."""

        if self.lock_button == 0:

            self.start_record = 1
            self.lock_button = 1
            if not self.wave_output_filename:
                self.wave_output_filename = os.path.join(os.getcwd(), "speech.wav")
                print(f"No path set. Using default: {self.wave_output_filename}")
            else:
                print(f"Saving to: {self.wave_output_filename}")

            self.wavfile = wave.open(self.wave_output_filename, 'wb')
            self.wavfile.setnchannels(2)
            self.wavfile.setsampwidth(2)
            self.wavfile.setframerate(self.sample_rate)

            print("Start recording...")
            self.audio_handle = pyaudio.PyAudio()
            self.counts_frames = 0
            self.draw_lock = 0

            def callback_streamin(in_data, frame_count, time_info, status):
                self.draw_lock = 1
                data_fr = np.frombuffer(in_data, dtype=np.int16).reshape(-1, 1) / 32768.0
                start = self.frame_size * self.counts_frames

                self.data_buffer[start:start + self.frame_size, 0] = data_fr[:, 0]

                if self.proc_st:
                    processed = self.proc_st(data_fr)
                    self.data_buffer[start:start + self.frame_size, 1] = processed[:, 0]

                else:
                    self.data_buffer[start:start + self.frame_size, 1] = 0.0

                int16_data = (np.clip(self.data_buffer[start:start + self.frame_size], -1.0, 1.0) * 32767).astype(np.int16)
                self.wavfile.writeframes(int16_data.tobytes())

                if self.start_record == 1:
                    if self.counts_frames == (self.num_blks - 1):
                        if self.non_stop:
                            self.data_buffer *= 0

                            ret = (in_data, pyaudio.paContinue)
                        else:
                            self.start_record = 0
                            ret = (in_data, pyaudio.paAbort)
                    else:
                        ret = (in_data, pyaudio.paContinue)
                else:
                    ret = (in_data, pyaudio.paAbort)

                self.counts_frames = (self.counts_frames + 1) % self.num_blks
                self.draw_lock = 0
                return ret

            self.stream = self.audio_handle.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.frame_size,
                stream_callback=callback_streamin
            )

            self.data_buffer *= 0
            self.line_data.set_data(self.const_data_buffer, self.data_buffer[:, 0])
            self.line_trig.set_data(self.const_data_buffer, self.data_buffer[:, 1])

            self.stream.start_stream()
            while self.stream.is_active():
                if self.draw_lock == 0:
                    self.line_data.set_data(self.const_data_buffer, self.data_buffer[:, 0])
                    self.line_trig.set_data(self.const_data_buffer, self.data_buffer[:, 1])
                    ending = self.frame_time_record * self.counts_frames
                    self.line_stop.set_data([ending, ending], LINE_MINMAX)
                    self.line_stop_vad.set_data([ending, ending], LINE_MINMAX)
                plt.pause(self.frame_time_replay)

            self.line_stop.set_data([0, 0], LINE_MINMAX)
            self.line_stop_vad.set_data([0, 0], LINE_MINMAX)
            self.stream.stop_stream()
            self.stream.close()
            self.audio_handle.terminate()
            self.wavfile.close()
            self.lock_button = 0
            print("Recording complete.")

        if hasattr(event, "inaxes") and event.inaxes is not None:
            event.inaxes.figure.canvas.draw_idle()
    

    def _playsound(self, wavname):
        event_obj = threading.Event()
        replay_buffer = np.zeros((self.sample_rate * self.record_seconds, ), dtype=float)
        with sf.SoundFile(wavname) as wavefile:
            def callback_streamout(outdata, framesize, time, status):
                self.draw_lock = 1
                data = wavefile.buffer_read(framesize, dtype='float32')
                data = np.frombuffer(data, dtype='float32').reshape(-1, wavefile.channels)

                if data.shape[0] == 0:
                    raise sd.CallbackStop

                channel_data = data[:, self.replay_channel % wavefile.channels]

                # Fill remaining buffer with zeros if data is too short
                if channel_data.shape[0] < framesize:
                    padded = np.zeros(framesize, dtype='float32')
                    padded[:channel_data.shape[0]] = channel_data
                    outdata[:] = padded.reshape(-1, 1)
                    raise sd.CallbackStop  # Final frame
                else:
                    outdata[:] = channel_data.reshape(-1, 1)
                
                replay_buffer[self.counts_frames * self.frame_size:(self.counts_frames + 1) * self.frame_size] = channel_data
                

                if self.replay_channel == 0:
                    line_draw = self.line_data
                else:
                    line_draw = self.line_trig
                line_draw.set_data(self.const_data_buffer, replay_buffer)
                

                self.counts_frames += 1
                if self.counts_frames == self.num_blks:
                    self.counts_frames = 0
                    self.line_data.set_data(self.const_data_buffer, replay_buffer*0)
                    
                    self.line_trig.set_data(self.const_data_buffer, replay_buffer*0)
                    
                
                self.draw_lock = 0

            self.counts_frames = 0
            self.data_buffer *= 0
            stream = sd.OutputStream(
                samplerate=wavefile.samplerate,
                channels=1,
                callback=callback_streamout,
                blocksize=self.frame_size,
                dtype='float32',
                finished_callback=event_obj.set)

            if self.replay_channel == 0:
                lineStop_selected = self.line_stop
            else:
                lineStop_selected = self.line_stop_vad

            with stream:
                while not event_obj.is_set():
                    if self.draw_lock == 0:
                        ending = self.frame_time_record * (self.counts_frames + 1)
                        lineStop_selected.set_data([ending, ending], LINE_MINMAX)
                    plt.pause(self.frame_time_replay)
                lineStop_selected.set_data([0, 0], LINE_MINMAX)


    def callback_replay(self, event):
        if self.lock_button == 0:
            self.lock_button = 1
            if os.path.exists(self.wave_output_filename):
                print(f'Start playback (channel {self.replay_channel})...')
                self._playsound(self.wave_output_filename)
                print('Playback finished.')
            self.lock_button = 0

        if hasattr(event, "inaxes") and event.inaxes is not None:
            event.inaxes.figure.canvas.draw_idle()

if __name__ == "__main__":
    aud_handle = AudioShowClass(
        record_seconds=15,
        non_stop=True,
        proc_st=simple_trigger_processor_vad
    )
