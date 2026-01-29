"""
Audio Viewer for the audio data from EVB - Dual Channel Visualization
Includes Correlation Score Reporting for Enrolled Users
"""
import os
import argparse
import sys
import wave
import multiprocessing
from multiprocessing import Process, Array, Lock
import time
import erpc
from . import GenericDataOperations_EvbToPc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import scipy.io.wavfile as wavfile
from .usb_utils import find_tinyusb_port

# --- Constants ---
FRAMES_TO_SHOW  = 500
SAMPLING_RATE   = 16000
HOP_SIZE        = 160
TEST_PHASE      = 1
ENROLL_PHASE    = 0
MAX_NUM_PPLS_ENROLL = 5

PC_INFO_ID = {
    "is_record"             : 0,
    "id_enroll_ppl"         : 1,
    "total_enroll_ppls"     : 2,
    "enroll_state"          : 3,
    "enroll_success"        : 4,
    "update_result"         : 5
}

EVB_INFO_ID = {
    "acc_utterances_enroll" : 0,
    "is_displayID"          : 1,
    "corr0"                 : 2,
    "corr1"                 : 3,
    "corr2"                 : 4,
    "corr3"                 : 5,
    "corr4"                 : 6
}

class DataServiceClass:
    def __init__(self, databuf1, databuf2, wavout, lock, pc_info, cyc_count, evb_info):
        self.cyc_count = cyc_count
        self.wavefile = None
        self.wavename = wavout
        self.databuf1 = databuf1 
        self.databuf2 = databuf2 
        self.lock = lock
        self.pc_info = pc_info
        self.evb_info = evb_info

    def wavefile_init(self, wavename):
        fldr = 'audio_result'
        os.makedirs(fldr, exist_ok=True)
        wf = wave.open(f'{fldr}/{wavename}', 'wb')
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        return wf

    def ns_rpc_data_sendBlockToPC(self, pcmBlock):
        self.lock.acquire()
        is_record = self.pc_info[PC_INFO_ID["is_record"]]
        self.lock.release()

        if is_record == 0:
            if self.wavefile:
                self.wavefile.close()
                try:
                    samplerate, sig = wavfile.read("audio_result/audio.wav")
                    wavfile.write("audio_result/audio_raw.wav", samplerate, sig[:,0].astype(np.int16))
                    wavfile.write("audio_result/audio_debug.wav", samplerate, sig[:,1].astype(np.int16))
                except Exception as e:
                    print(f"Wav conversion error: {e}")
                self.wavefile = None
                print('Stop recording. Files saved in audio_result/')
        else:
            if not self.wavefile:
                print('Start recording')
                self.lock.acquire()
                self.cyc_count[0] = 0
                self.lock.release()
                self.wavefile = self.wavefile_init(self.wavename)

            if (pcmBlock.cmd == GenericDataOperations_EvbToPc.common.command.write_cmd) \
                     and (pcmBlock.description == "Audio16bPCM_to_WAV"):

                data = np.frombuffer(pcmBlock.buffer, dtype=np.int16).copy()
                
                # Metadata extraction
                acc_utterances_enroll = data[HOP_SIZE*2]
                is_displayID = data[HOP_SIZE*2+1]

                self.lock.acquire()
                self.evb_info[EVB_INFO_ID["acc_utterances_enroll"]] = acc_utterances_enroll
                if is_displayID == 1:
                    self.pc_info[PC_INFO_ID["update_result"]] = 1
                    self.evb_info[EVB_INFO_ID["is_displayID"]] = is_displayID
                    total_enroll_ppls = self.pc_info[PC_INFO_ID["total_enroll_ppls"]]
                    # Fill correlation scores into evb_info
                    for i in range(total_enroll_ppls):
                        self.evb_info[i+2] = data[HOP_SIZE*2+2+i]
                
                enroll_state = self.pc_info[PC_INFO_ID["enroll_state"]]
                if enroll_state == ENROLL_PHASE and acc_utterances_enroll == 4:
                    self.pc_info[PC_INFO_ID["enroll_success"]] = 1
                    self.pc_info[PC_INFO_ID["is_record"]] = 0
                self.lock.release()

                # Save raw PCM
                audio_data = data[:HOP_SIZE*2].reshape((2, HOP_SIZE)).T.flatten()
                self.wavefile.writeframesraw(audio_data.tobytes())

                # Update Visual Buffers
                fdata1 = data[:HOP_SIZE].astype(np.float32) / 32768.0
                fdata2 = data[HOP_SIZE:2*HOP_SIZE].astype(np.float32) / 32768.0

                self.lock.acquire()
                curr_idx = self.cyc_count[0]
                start = curr_idx * HOP_SIZE
                self.databuf1[start:start+HOP_SIZE] = fdata1
                self.databuf2[start:start+HOP_SIZE] = fdata2
                self.cyc_count[0] = (curr_idx + 1) % FRAMES_TO_SHOW
                self.lock.release()

        return 0

    def ns_rpc_data_fetchBlockFromPC(self, block): return 0

    def ns_rpc_data_computeOnPC(self, in_block, IsRecordBlock):
        if (in_block.cmd == GenericDataOperations_EvbToPc.common.command.extract_cmd) and (
            in_block.description == "CalculateMFCC_Please"):
            self.lock.acquire()
            data2pc = [
                self.pc_info[PC_INFO_ID["is_record"]],
                self.pc_info[PC_INFO_ID["id_enroll_ppl"]],
                self.pc_info[PC_INFO_ID["total_enroll_ppls"]],
                self.pc_info[PC_INFO_ID["enroll_state"]]
            ]
            self.lock.release()
            IsRecordBlock.value = GenericDataOperations_EvbToPc.common.dataBlock(
                description="*\0",
                dType=GenericDataOperations_EvbToPc.common.dataType.uint8_e,
                cmd=GenericDataOperations_EvbToPc.common.command.generic_cmd,
                buffer=bytearray(data2pc),
                length=len(data2pc),
            )
        return 0

class VisualDataClass:
    def __init__(self, databuf1, databuf2, lock, pc_info, event_stop, cyc_count, evb_info, thres_nnid=0.8):
        self.enroll_names = {}
        self.total_enroll_ppls = 0
        self.databuf1 = databuf1
        self.databuf2 = databuf2
        self.lock = lock
        self.pc_info = pc_info
        self.event_stop = event_stop
        self.cyc_count = cyc_count
        self.evb_info = evb_info
        self.thres_nnid = thres_nnid

        secs2show = FRAMES_TO_SHOW * HOP_SIZE / SAMPLING_RATE
        self.xdata = np.arange(FRAMES_TO_SHOW * HOP_SIZE) / SAMPLING_RATE
        
        # --- UI Layout Setup ---
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        self.fig.canvas.mpl_connect('close_event', self.handle_close)
        
        plt.subplots_adjust(bottom=0.25, hspace=0.35)
        self.title_handle = self.fig.suptitle("NeuralSPOT Audio Viewer: Dual Channel", fontsize=14, y=0.96)

        self.line1, = self.ax1.plot(self.xdata, [0]*len(self.xdata), lw=0.5, color='blue')
        self.line2, = self.ax2.plot(self.xdata, [0]*len(self.xdata), lw=0.5, color='red')
        
        self.ax1.set_ylabel("Ch 0 (Raw)")
        self.ax2.set_ylabel("Ch 1 (Debug)")
        for ax in [self.ax1, self.ax2]:
            ax.set_ylim([-1.1, 1.1])
            ax.set_xlim((0, secs2show))
            ax.grid(True, alpha=0.2)
        self.ax2.set_xlabel('Time (Seconds)')

        # --- Dashboard Controls ---
        axbox = plt.axes([0.15, 0.12, 0.25, 0.05])
        self.enroll_box = TextBox(axbox, 'Name: ', initial="")
        
        ax_enroll = plt.axes([0.45, 0.12, 0.15, 0.05])
        self.btn_enroll = Button(ax_enroll, 'Enroll', color='whitesmoke', hovercolor='lightgreen')
        self.btn_enroll.on_clicked(self.callback_enroll)

        ax_test = plt.axes([0.15, 0.04, 0.15, 0.05])
        self.btn_test = Button(ax_test, 'Test Mode', color='whitesmoke', hovercolor='lightblue')
        self.btn_test.on_clicked(self.callback_test)

        ax_stop = plt.axes([0.45, 0.04, 0.15, 0.05])
        self.btn_stop = Button(ax_stop, 'Stop', color='whitesmoke', hovercolor='tomato')
        self.btn_stop.on_clicked(self.callback_recordstop)

        self.txt_status = self.fig.text(0.65, 0.08, f"Threshold: {self.thres_nnid}\nRegistered: 0", 
                                       fontsize=10, bbox=dict(facecolor='none', edgecolor='gray', pad=5))

        plt.show()

    def update_plots(self):
        self.lock.acquire()
        cyc = self.cyc_count[0]
        d1 = np.array(self.databuf1[:])
        d2 = np.array(self.databuf2[:])
        self.lock.release()
        
        clean_d1 = np.zeros_like(d1)
        clean_d2 = np.zeros_like(d2)
        clean_d1[:cyc*HOP_SIZE] = d1[:cyc*HOP_SIZE]
        clean_d2[:cyc*HOP_SIZE] = d2[:cyc*HOP_SIZE]
        
        self.line1.set_ydata(clean_d1)
        self.line2.set_ydata(clean_d2)

    def callback_enroll(self, event):
        name = self.enroll_box.text.strip()
        if not name: return

        self.lock.acquire()
        if self.pc_info[PC_INFO_ID["is_record"]] == 0:
            if name not in self.enroll_names:
                self.enroll_names[name] = self.total_enroll_ppls
                self.total_enroll_ppls += 1
            
            self.pc_info[PC_INFO_ID["is_record"]] = 1
            self.pc_info[PC_INFO_ID["id_enroll_ppl"]] = self.enroll_names[name]
            self.pc_info[PC_INFO_ID["total_enroll_ppls"]] = self.total_enroll_ppls
            self.pc_info[PC_INFO_ID["enroll_state"]] = ENROLL_PHASE
            self.pc_info[PC_INFO_ID["enroll_success"]] = 0
            self.lock.release()

            self.txt_status.set_text(f"Threshold: {self.thres_nnid}\nRegistered: {self.total_enroll_ppls}")
            
            while True:
                self.update_plots()
                self.lock.acquire()
                acc = self.evb_info[EVB_INFO_ID["acc_utterances_enroll"]]
                active = self.pc_info[PC_INFO_ID["is_record"]]
                self.lock.release()
                self.title_handle.set_text(f"Enrolling {name}: {acc}/4 phrases")
                plt.pause(0.01)
                if active == 0: break

    def callback_test(self, event):
        if not self.enroll_names: 
            print("Console: No users enrolled yet.")
            return

        self.lock.acquire()
        if self.pc_info[PC_INFO_ID["is_record"]] == 0:
            self.pc_info[PC_INFO_ID["is_record"]] = 1
            self.pc_info[PC_INFO_ID["enroll_state"]] = TEST_PHASE
            self.lock.release()
            
            id2name = {v: k for k, v in self.enroll_names.items()}
            print("\n--- TEST MODE ACTIVE ---")

            while True:
                self.update_plots()
                self.lock.acquire()
                
                if self.pc_info[PC_INFO_ID["update_result"]] == 1:
                    count = self.pc_info[PC_INFO_ID["total_enroll_ppls"]]
                    scores = []
                    
                    print("-" * 30)
                    for i in range(count):
                        # Extract and normalize score (0.0 to 1.0)
                        raw_val = self.evb_info[i + 2]
                        conf = float(raw_val) / 32768.0
                        scores.append(conf)
                        print(f"User: {id2name[i]:<10} | Correlation: {conf:.4f}")

                    best_id = np.argmax(scores)
                    best_conf = scores[best_id]
                    
                    if best_conf > self.thres_nnid:
                        self.title_handle.set_text(f"Verified: {id2name[best_id]} ({best_conf:.2f})")
                    else:
                        self.title_handle.set_text(f"Unknown (Best: {best_conf:.2f})")
                    
                    self.pc_info[PC_INFO_ID["update_result"]] = 0
                
                active = self.pc_info[PC_INFO_ID["is_record"]]
                self.lock.release()
                plt.pause(0.01)
                if active == 0: break

    def callback_recordstop(self, event):
        self.lock.acquire()
        self.pc_info[PC_INFO_ID["is_record"]] = 0
        self.lock.release()

    def handle_close(self, event):
        self.lock.acquire()
        self.pc_info[PC_INFO_ID["is_record"]] = 0
        self.lock.release()
        self.event_stop.set()

def target_proc_draw(db1, db2, lock, pc_info, event_stop, cyc_count, evb_info, thres):
    VisualDataClass(db1, db2, lock, pc_info, event_stop, cyc_count, evb_info, thres)

def target_proc_evb2pc(tty, baud, db1, db2, out, lock, pc_info, cyc_count, evb_info):
    try:
        transport = erpc.transport.SerialTransport(tty, int(baud))
        handler = DataServiceClass(db1, db2, out, lock, pc_info, cyc_count, evb_info)
        service = GenericDataOperations_EvbToPc.server.evb_to_pcService(handler)
        server = erpc.simple_server.SimpleServer(transport, erpc.basic_codec.BasicCodec)
        server.add_service(service)
        server.run()
    except Exception as e: 
        print(f"Server Error: {e}")

def main(args):
    event_stop = multiprocessing.Event()
    lock = Lock()
    db1 = Array('d', FRAMES_TO_SHOW * HOP_SIZE)
    db2 = Array('d', FRAMES_TO_SHOW * HOP_SIZE)
    pc_info = Array('i', [0]*6)
    evb_info = Array('i', [0]*7)
    cyc_count = Array('i', [0])
    
    tty = find_tinyusb_port()
    if not tty: 
        print("Error: Could not find EVB Port.")
        return

    p_draw = Process(target=target_proc_draw, args=(db1, db2, lock, pc_info, event_stop, cyc_count, evb_info, args.thres_nnid))
    p_rpc = Process(target=target_proc_evb2pc, args=(tty, args.baud, db1, db2, args.out, lock, pc_info, cyc_count, evb_info))
    
    p_draw.start()
    p_rpc.start()
    
    try:
        while not event_stop.is_set():
            time.sleep(0.1)
    finally:
        p_draw.terminate()
        p_rpc.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--baud", default="115200")
    parser.add_argument("-th", "--thres_nnid", default=0.8, type=float)
    parser.add_argument("-o", "--out", default="audio.wav")
    main(parser.parse_args())