"""
Audio Viewer for the audio data from EVB
"""
import os
import argparse
import sys
import wave
import multiprocessing
from multiprocessing import Process, Array, Lock
import time
import erpc
import GenericDataOperations_EvbToPc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import scipy.io.wavfile as wavfile

# Define the RPC service handlers - one for each EVB-to-PC RPC function
FRAMES_TO_SHOW  = 500
SAMPLING_RATE   = 16000
HOP_SIZE        = 160
TEST_PHASE      = 1
ENROLL_PHASE    = 0
MAX_NUM_PPLS_ENROLL = 5
PC_INFO_ID={
    "is_record"             :0,
    "id_enroll_ppl"         :1,
    "total_enroll_ppls"     :2,
    "enroll_state"          :3,
    "enroll_success"        :4,
    "update_result"         :5}

EVB_INFO_ID={
    "acc_utterances_enroll" :0,
    "is_displayID"          :1,
    "corr0"                 :2,
    "corr1"                 :3,
    "corr2"                 :4,
    "corr3"                 :5,
    "corr4"                 :6}

class DataServiceClass:
    """
    Capture Audio data: EVB->PC
    """
    def __init__(self, databuf, wavout, lock, pc_info, cyc_count, evb_info):
        self.cyc_count      = cyc_count
        self.wavefile       = None
        self.wavename       = wavout
        self.databuf        = databuf
        self.lock           = lock
        self.pc_info      = pc_info
        self.evb_info     = evb_info

    def wavefile_init(self, wavename):
        """
        wavefile initialization
        """
        fldr = 'audio_result'
        os.makedirs(fldr, exist_ok=True)
        wavefile = wave.open(f'{fldr}/{wavename}', 'wb')
        wavefile.setnchannels(2)
        wavefile.setsampwidth(2)
        wavefile.setframerate(16000)
        return wavefile

    def ns_rpc_data_sendBlockToPC(self, pcmBlock): # pylint: disable=invalid-name
        """
        callback function that data sent from EVB to PC.
        """
        self.lock.acquire()
        is_record = self.pc_info[PC_INFO_ID["is_record"]]
        self.lock.release()
        if is_record == 0:
            if self.wavefile:
                self.wavefile.close()

                samplerate, sig = wavfile.read("audio_result/audio.wav")
                sig1 = sig[:,0].flatten()
                wavfile.write("audio_result/audio_raw.wav", samplerate, sig1.astype(np.int16))

                sig2 = sig[:,1].flatten()
                wavfile.write("audio_result/audio_debug.wav", samplerate, sig2.astype(np.int16))

                self.wavefile = None
                print('Stop recording')
        else:
            # The data 'block' (in C) is defined below:
            # static char msg_store[30] = "Audio16bPCM_to_WAV";

            # // Block sent to PC
            # static dataBlock outBlock = {
            #     .length = SAMPLES_IN_FRAME * sizeof(int16_t),
            #     .dType = uint8_e,
            #     .description = msg_store,
            #     .cmd = write_cmd,
            #     .buffer = {.data = (uint8_t *)in16AudioDataBuffer, // point this to audio buffer # pylint: disable=line-too-long
            #             .dataLength = SAMPLES_IN_FRAME * sizeof(int16_t)}};

            if self.wavefile:
                self.lock.acquire()
                cyc_count = self.cyc_count[0]
                self.lock.release()
            else:
                print('Start recording')
                cyc_count = 0

                self.lock.acquire()
                self.cyc_count[0] = cyc_count
                self.lock.release()

                self.wavefile = self.wavefile_init(self.wavename)

            if (pcmBlock.cmd == GenericDataOperations_EvbToPc.common.command.write_cmd) \
                     and (pcmBlock.description == "Audio16bPCM_to_WAV"):

                self.lock.acquire()
                data = np.frombuffer(pcmBlock.buffer, dtype=np.int16).copy()
                self.lock.release()

                acc_utterances_enroll = data[HOP_SIZE*2]
                is_displayID = data[HOP_SIZE*2+1]
                self.lock.acquire()
                self.evb_info[0] = acc_utterances_enroll
                if is_displayID==1:
                    self.pc_info[PC_INFO_ID["update_result"]] = 1
                    self.evb_info[EVB_INFO_ID["is_displayID"]] = is_displayID
                enroll_state = self.pc_info[PC_INFO_ID["enroll_state"]]

                if is_displayID == 1:
                    total_enroll_ppls  = self.pc_info[PC_INFO_ID["total_enroll_ppls"]]
                    for i in range(total_enroll_ppls):
                        self.evb_info[i+2] = data[HOP_SIZE*2+2+i]
                self.lock.release()
                # print(f"enroll_state = {enroll_state}, acc_utterances_enroll = {acc_utterances_enroll}, is_displayID = {is_displayID}") # pylint: disable=line-too-long
                if enroll_state == ENROLL_PHASE:
                    if acc_utterances_enroll==4:
                        self.lock.acquire()
                        self.pc_info[PC_INFO_ID["enroll_success"]]  = 1
                        self.pc_info[PC_INFO_ID["is_record"]] = 0
                        self.lock.release()

                data = data[:HOP_SIZE*2]
                data = data.reshape((2, HOP_SIZE)).T.flatten()
                self.wavefile.writeframesraw(data.tobytes())

                # Data is a 16 bit PCM sample
                self.lock.acquire()
                fdata = np.frombuffer(pcmBlock.buffer, dtype=np.int16).copy() / 32768.0
                self.lock.release()

                start = cyc_count * HOP_SIZE

                self.lock.acquire()
                self.databuf[start:start+HOP_SIZE] = fdata[:HOP_SIZE]
                self.lock.release()

                cyc_count = (cyc_count+1) % FRAMES_TO_SHOW

                self.lock.acquire()
                self.cyc_count[0] = cyc_count
                self.lock.release()

        sys.stdout.flush()

        return 0

    def ns_rpc_data_fetchBlockFromPC(self, block): # pylint: disable=invalid-name, unused-argument
        """
        callback function that Data fetching
        """
        sys.stdout.flush()
        return 0

    def ns_rpc_data_computeOnPC( # pylint: disable=invalid-name
            self,
            in_block,       # like a request block from EVB
            IsRecordBlock):  # send the result_block to EVB
        """
        callback function that sending result_block to EVB
            that indicating to record or stop
        """
        if (in_block.cmd == GenericDataOperations_EvbToPc.common.command.extract_cmd) and (
            in_block.description == "CalculateMFCC_Please"):
            self.lock.acquire()
            a0 = self.pc_info[PC_INFO_ID["is_record"]]
            a1 = self.pc_info[PC_INFO_ID["id_enroll_ppl"]]
            a2 = self.pc_info[PC_INFO_ID["total_enroll_ppls"]]
            a3 = self.pc_info[PC_INFO_ID["enroll_state"]]
            self.lock.release()
            data2pc = [a0, a1, a2, a3]
            IsRecordBlock.value = GenericDataOperations_EvbToPc.common.dataBlock(
                description ="*\0",
                dType       = GenericDataOperations_EvbToPc.common.dataType.uint8_e,
                cmd         = GenericDataOperations_EvbToPc.common.command.generic_cmd,
                buffer      = bytearray(data2pc),
                length      = len(data2pc),
            )
        sys.stdout.flush()
        return 0

class VisualDataClass:
    """
    Visual the audio data from EVB
    """
    def __init__(
            self,
            databuf,
            lock,
            pc_info,
            event_stop,
            cyc_count,
            evb_info,
            thres_nnid = 0.8):
        self.enroll_names={}
        self.total_enroll_ppls = 0

        self.databuf = databuf
        self.lock    = lock
        self.pc_info = pc_info
        self.event_stop = event_stop
        self.cyc_count = cyc_count
        self.evb_info = evb_info
        self.thres_nnid = thres_nnid
        secs2show = FRAMES_TO_SHOW * HOP_SIZE/SAMPLING_RATE
        self.xdata = np.arange(FRAMES_TO_SHOW * HOP_SIZE) / SAMPLING_RATE
        self.fig, self.ax_handle = plt.subplots()
        self.fig.canvas.mpl_connect('close_event', self.handle_close)
        plt.subplots_adjust(bottom=0.35)
        self.title_handle = plt.title("Click 'enroll' button to start the enrollment")
        self.text_thres = plt.text(0, -2.5, f"Threshold={self.thres_nnid}")
        self.text_enroll_info = plt.text(4, -2, "No enrollment info")
        self.name_current_enroll = ""
        self.lock.acquire()
        np_databuf = databuf[0:]
        self.lock.release()

        self.line_data, = self.ax_handle.plot(self.xdata, np_databuf, lw=0.5, color = 'blue')
        plt.ylim([-1.1,1.1])
        self.ax_handle.set_xlim((0, secs2show))
        self.ax_handle.set_xlabel('Time (Seconds)')
        plt.plot(
            [0, secs2show],
            [1, 1],
            color='black',
            lw=1)
        plt.plot(
            [0, secs2show],
            [-1, -1],
            color='black',
            lw=1)
        # making buttons
        def make_button(pos, name, callback_func):
            ax_button = plt.axes(pos)
            button = Button(
                        ax_button,
                        name,
                        color = 'w',
                        hovercolor = 'aliceblue')
            button.label.set_fontsize(16)
            button.on_clicked(callback_func)
            return button
        self.wavfile = None
        self.button_stop = make_button(
                            [0.35, 0.05, 0.14, 0.075],
                            'stop',
                            self.callback_recordstop)
        self.button_enroll = make_button(
                            [0.5, 0.15, 0.14, 0.075],
                            'enroll',
                            self.callback_enroll)
        self.button_test = make_button(
                            [0.5, 0.05, 0.14, 0.075],
                            'test',
                            self.callback_test)

        axbox = plt.axes([0.35, 0.15, 0.14, 0.075])
        self.enroll_box = TextBox(
                    axbox,
                    'Input your Name->',
                    initial="")
        plt.show()

    def callback_enroll(self, event):
        """
        for enroll text box
        """
        if self.enroll_box.text == "":
            self.title_handle.set_text("Input your name to enroll")
        else:
            text = self.enroll_box.text
            self.lock.acquire()
            is_record = self.pc_info[PC_INFO_ID["is_record"]]
            self.lock.release()
            if is_record == 0:
                if text in self.enroll_names:
                    print(f"Name {text} enrolled as id = {self.enroll_names[text]}")
                else:
                    self.enroll_names[text] = self.total_enroll_ppls
                    print(f"Name {text} enrolled as id = {self.total_enroll_ppls}")
                    self.total_enroll_ppls += 1
                self.name_current_enroll = text
                if self.total_enroll_ppls > MAX_NUM_PPLS_ENROLL:
                    print(f"Max number of ppl to enroll is {MAX_NUM_PPLS_ENROLL}")
                else:
                    self.lock.acquire()
                    self.pc_info[PC_INFO_ID["is_record"]]       = 1
                    self.pc_info[PC_INFO_ID["id_enroll_ppl"]]   = self.enroll_names[self.name_current_enroll] # pylint: disable=line-too-long
                    self.pc_info[PC_INFO_ID["total_enroll_ppls"]]      = self.total_enroll_ppls
                    self.pc_info[PC_INFO_ID["enroll_state"]]    = ENROLL_PHASE
                    self.pc_info[PC_INFO_ID["enroll_success"]]  = 0
                    self.lock.release()
                    while 1:
                        self.lock.acquire()
                        cyc_count = self.cyc_count[0]
                        np_databuf = self.databuf[0:]
                        acc_utterances_enroll = self.evb_info[EVB_INFO_ID["acc_utterances_enroll"]]
                        self.lock.release()

                        zeros_tail = [0.0] * (HOP_SIZE * (FRAMES_TO_SHOW - cyc_count))
                        np_databuf = np_databuf[:HOP_SIZE*cyc_count] + zeros_tail
                        self.line_data.set_data(self.xdata, np_databuf)

                        self.title_handle.set_text(f"{self.name_current_enroll}: you have {acc_utterances_enroll} / 4 utterances in enrollment. \nPlease say something") # pylint: disable=line-too-long

                        self.lock.acquire()
                        enroll_success = self.pc_info[PC_INFO_ID["enroll_success"]]
                        is_record = self.pc_info[PC_INFO_ID["is_record"]]
                        self.lock.release()

                        if is_record==0:
                            if enroll_success == 1:
                                self.title_handle.set_text(f"{self.name_current_enroll}'s enrollment succeeded")  # pylint: disable=line-too-long
                                info = "Enrollment info:\n"
                                for key, val in self.enroll_names.items():
                                    info+=f"{val} : {key}\n"
                                self.text_enroll_info.set_text(info)
                            else:
                                self.title_handle.set_text(f"{self.name_current_enroll}'s enrollment failed")   # pylint: disable=line-too-long

                        plt.pause(0.05)
                        if is_record == 0:
                            break
        if event.inaxes is not None:
            event.inaxes.figure.canvas.draw_idle()

    def handle_close(self, event): # pylint: disable=unused-argument
        """
        Finish everything when you close your plot
        """
        self.lock.acquire()
        self.pc_info[PC_INFO_ID["is_record"]] = 0
        self.lock.release()
        print('Window close')
        time.sleep(0.05)
        self.event_stop.set() # let main function know program should be terminated now

    def callback_recordstop(self, event):
        """
        for stop button
        """
        self.lock.acquire()
        is_record = self.pc_info[PC_INFO_ID["is_record"]]
        enroll_state = self.pc_info[PC_INFO_ID["enroll_state"]]
        enroll_success = self.pc_info[PC_INFO_ID["enroll_success"]]
        self.lock.release()
        print(f"enroll_success = {enroll_success}")
        if enroll_state == ENROLL_PHASE:
            if is_record == 1:
                self.lock.acquire()
                self.pc_info[PC_INFO_ID["is_record"]] = 0
                enroll_success = self.pc_info[PC_INFO_ID["enroll_success"]]
                self.lock.release()
                if enroll_success == 0:
                    del self.enroll_names[self.name_current_enroll]
                    self.total_enroll_ppls -= 1
                    self.lock.acquire()
                    self.pc_info[PC_INFO_ID["total_enroll_ppls"]] = self.total_enroll_ppls
                    self.lock.release()
                else:
                    self.title_handle.set_text("Click 'enroll' button to start the enrollment")
        else: # TEST_PHASE
            if is_record==1:
                self.lock.acquire()
                self.pc_info[PC_INFO_ID["is_record"]] = 0
                self.lock.release()

        if event.inaxes is not None:
            event.inaxes.figure.canvas.draw_idle()

    def callback_test(self, event):
        """
        for record button
        """
        if len(self.enroll_names) == 0:
            print("No enrollment info")
            return 0
        self.lock.acquire()
        is_record = self.pc_info[PC_INFO_ID["is_record"]]
        self.lock.release()
        if is_record == 0:
            self.lock.acquire()
            self.pc_info[PC_INFO_ID["is_record"]]       = 1
            self.pc_info[PC_INFO_ID["enroll_state"]]    = TEST_PHASE
            self.lock.release()
            corr = np.zeros((MAX_NUM_PPLS_ENROLL,), dtype=np.int32)
            enroll_id2name = dict((v,k) for k,v in self.enroll_names.items())
            while 1:
                self.lock.acquire()
                cyc_count = self.cyc_count[0]
                np_databuf = self.databuf[0:]
                self.lock.release()

                zeros_tail = [0.0] * (HOP_SIZE * (FRAMES_TO_SHOW - cyc_count))
                np_databuf = np_databuf[:HOP_SIZE*cyc_count] + zeros_tail
                self.line_data.set_data(self.xdata, np_databuf)

                self.lock.acquire()
                update_result = self.pc_info[PC_INFO_ID["update_result"]]
                if update_result == 1:
                    total_enroll_ppls  = self.pc_info[PC_INFO_ID["total_enroll_ppls"]]
                    for i in range(total_enroll_ppls):
                        corr[i] = self.evb_info[i+2]
                self.lock.release()
                if update_result == 1:
                    id_max_corr = np.argmax(corr)
                    max_corr = float(corr[id_max_corr]) / 32768.0
                    if max_corr > self.thres_nnid:
                        self.title_handle.set_text(f"{enroll_id2name[id_max_corr]} is verified") # pylint: disable=line-too-long
                    else:
                        self.title_handle.set_text(f"Unknown: corr = {max_corr:.2f}")
                    info = "Testing info:\n"
                    for i, (key, val) in enumerate(self.enroll_names.items()):
                        corr_i = float(corr[i]/32768.0)
                        info+=f"{val} : {key}--corr={corr_i:.2f}\n"
                    self.text_enroll_info.set_text(info)
                self.lock.acquire()
                self.pc_info[PC_INFO_ID["update_result"]] = 0
                self.lock.release()

                plt.pause(0.05)
                self.lock.acquire()
                is_record = self.pc_info[PC_INFO_ID["is_record"]]
                self.lock.release()
                if is_record == 0:
                    break
        if event.inaxes is not None:
            event.inaxes.figure.canvas.draw_idle()

def target_proc_draw(databuf, lock, recording, event_stop, cyc_count, enroll_ind, thres_nnid=0.8):
    """
    one of multiprocesses: draw
    """
    VisualDataClass(databuf, lock, recording, event_stop, cyc_count, enroll_ind, thres_nnid)

def target_proc_evb2pc(tty, baud, databuf, wavout, lock, is_record, cyc_count, enroll_ind):
    """
    one of multiprocesses: EVB sends data to PC
    """
    transport_evb2pc = erpc.transport.SerialTransport(tty, int(baud))
    handler = DataServiceClass(databuf, wavout, lock, is_record, cyc_count, enroll_ind)
    service = GenericDataOperations_EvbToPc.server.evb_to_pcService(handler)
    server = erpc.simple_server.SimpleServer(transport_evb2pc, erpc.basic_codec.BasicCodec)
    server.add_service(service)
    print("\r\nServer started - waiting for EVB to send an eRPC request")
    sys.stdout.flush()
    server.run()

def main(args):
    """
    main
    """
    event_stop = multiprocessing.Event()
    lock = Lock()
    databuf = Array('d', FRAMES_TO_SHOW * HOP_SIZE)
    """
    pc_info:
        0 : is_record indicator
        1 : id of ppl to enroll
        2 : total ppl enrolled
        3 : enroll_state
        4 : enroll_success
    """
    pc_info = Array('i', [0,0,0,0,0,0])
    evb_info = Array('i', [0,0,0,0,0,0,0]) # (acc_utterances_enroll, is_displayID, correlation_id[5])
    cyc_count = Array('i', [0])
    # we use two multiprocesses to handle real-time visualization and recording
    # 1. proc_draw   : to visualize
    # 2. proc_evb2pc : to capture data from evb and recording
    proc_draw   = Process(
                    target = target_proc_draw,
                    args   = (databuf,
                              lock,
                              pc_info,
                              event_stop,
                              cyc_count,
                              evb_info,
                              args.thres_nnid))
    proc_evb2pc = Process(
                    target = target_proc_evb2pc,
                    args   = (  args.tty,
                                args.baud,
                                databuf,
                                args.out,
                                lock,
                                pc_info,
                                cyc_count,
                                evb_info))
    proc_draw.start()
    proc_evb2pc.start()
    # monitor if program should be terminated
    while True:
        if event_stop.is_set():
            proc_draw.terminate()
            proc_evb2pc.terminate()
            #Terminating main process
            sys.exit(1)
        time.sleep(0.5)

if __name__ == "__main__":

    # parse cmd parameters
    argParser = argparse.ArgumentParser(description="NeuralSPOT GenericData RPC Demo")

    argParser.add_argument(
        "-w",
        "--tty",
        default = "/dev/serial/by-id/usb-TinyUSB_TinyUSB_Device_123457-if00", # "/dev/tty.usbmodem1234561"
        help    = "Serial device (default value is None)",
    )
    argParser.add_argument(
        "-B",
        "--baud",
        default = "115200",
        help    = "Baud (default value is 115200)"
    )

    argParser.add_argument(
        "-th",
        "--thres_nnid",
        default = 0.8,
        type    = float,
        help    = "threshold for nnid"
    )

    argParser.add_argument(
        "-o",
        "--out",
        default = "audio.wav",
        help    = "File where data will be written (default is audio.wav",
    )

    main(argParser.parse_args())
