''' prepare tfrecords data for KWS task '''
import logging
import random
import re
import multiprocessing
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from soundkit.utils.tf_basic_math import tf_log10_eps
from soundkit.utils.audio import pad_or_crop
from soundkit.defines import SKTaskParams
from soundkit.utils.feature_utils import FeatureExtractor
from soundkit.utils.basic_dsp import dc_remove
from soundkit.utils.download_api import corpus_download
from soundkit.utils.audio import (
        audio_read,
        random_load_audio_from_list,
        synthesize_audio_with_labels,
    )
from soundkit.utils.plot_api import plot_spectrograms
from soundkit.datasets import SKDatasetFactory
from .datasets import create_raw_tfrecord

logging.basicConfig(   
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
log = logging.getLogger(__name__)

class FeatMultiProcsClass(multiprocessing.Process):
    """
    A worker process for parallel feature extraction.
    """

    def __init__(
            self,
            proc_pid: int, # process id
            audio_type_list: dict[str, list],
            success_dict: dict,
            params: SKTaskParams,
            info: str = None,
            debug: bool = False,
            ):
        super().__init__()

        self.proc_pid = proc_pid
        self.speech_list = audio_type_list['speech']
        self.garb_list = audio_type_list['garbage']
        self.noise_list = audio_type_list['noise']
        self.reverb_list = audio_type_list['reverb']
        self.success_dict = success_dict
        self.params = params  # includes noise_files, snr_dbs, reverb_prob, output_dir, etc.
        self.info = info
        self.debug= debug

    def run(self):
        """
        Run the process to generate noisy and 
        clean audio files.
        """

        target_sample_rate = self.params.data['signal']["sampling_rate"]
        is_dc_removal = self.params.data['signal']["dc_removal"]
        revert_prob = self.params.data["reverb_prob"]
        path_tfrecord=self.params.data['path_tfrecord']
        target_length = int(self.params.data['target_length_in_secs'] * target_sample_rate)
        # print(f"[Process {self.proc_pid}] Started with {len(self.speech_list)} files.")
        target_frames_extension= self.params.data['target_frames_extension']
        hop_size = self.params.train['feature']['hop_size']
        len_garb_list = len(self.speech_list)
        for idx, sample in enumerate(tqdm(self.speech_list, desc=f"Processing {self.proc_pid}", unit="file", leave=False)):
            for k in range(4):
                
                if k == 0 or k==1:
                    # load clean speech
                    wavname, label = sample

                    clean = audio_read(
                        wavname,
                        sample_rate=target_sample_rate)

                    starts = np.array([seg['start'] for seg in label])
                    ends = np.array([seg['end'] for seg in label])

                    if k==1: # destroy kws speech
                        len_c = ends[0] - starts[0]
                        len_d = np.random.randint( len_c//2, len_c) # length destroy

                        rn = np.random.randint(0, 2) # random choice to destroy speech
                        rn1 = np.random.randint(0, 2) # random choice to remove target speech
                        if rn1==1:
                            idx1 = np.random.randint(0, len_garb_list)
                            wavname_g, labels = self.garb_list[idx1]
                            starts_g = np.array([seg['start'] for seg in labels])
                            ends_g = np.array([seg['end'] for seg in labels])

                            garb = audio_read(wavname_g, sample_rate=target_sample_rate)
                            garb, *_ = pad_or_crop(garb, target_length)

                            if len(ends_g) > 0:
                                garb = garb[starts_g[0]:ends_g[0]]
                            if len(garb) > len_d:
                                garb = garb[:len_d]
                            else:
                                garb = np.pad(garb, (0, len_d - len(garb)), mode='constant')
                        if rn == 0: # destroy the front of the keyword speech
                            clean[starts[0]: starts[0] + len_d] = 0.0
                            # if rn1 == 1: # remove target speech
                            #     clean[starts[0]: starts[0] + len_d] = garb
                        else: # destroy the back of the keyword speech
                            clean[ends[0] - len_d: ends[0]] = 0.0
                            if rn1 == 1: # remove target speech
                                clean[ends[0] - len_d: ends[0]] = garb
                    starts = ends -1
                    ends = starts + hop_size * target_frames_extension-1

                    pad_len_front = np.random.randint(hop_size*target_frames_extension, hop_size*target_frames_extension*5)
                    pad_len_back = np.random.randint(hop_size*target_frames_extension, hop_size*target_frames_extension*5)

                    clean = np.concatenate(
                        [np.zeros(pad_len_front), clean, np.zeros(pad_len_back)])

                    starts += pad_len_front
                    ends += pad_len_front

                    garbs = []
                    lens_garb = []
                    for g in range(2):
                        idx1 = np.random.randint(0, len_garb_list)
                        wavname_g, labels = self.garb_list[idx1]

                        garb = audio_read(wavname_g, sample_rate=target_sample_rate)
                        garb, *_ = pad_or_crop(garb, target_length)

                        len_garb = target_length >> 2

                        s = np.random.randint(0, len(garb) - len_garb)
                        gain = np.random.randint(0, 2)
                        garbs.append(garb[s:s+len_garb] * gain)

                        lens_garb.append(len_garb)
                    clean = np.concatenate([garbs[0], clean, garbs[1]])

                    if k == 0:
                        starts = starts + lens_garb[0]
                        ends = ends + lens_garb[0]
                    else: # k == 1: destroyed speech and removed target
                        starts = np.array([])
                        ends = np.array([])
                else:
                    wavname, labels = self.garb_list[(idx*3+k)%len_garb_list]
                    clean = audio_read(wavname, sample_rate=target_sample_rate)
                    starts = np.array([seg['start'] for seg in labels])
                    ends = np.array([seg['end'] for seg in labels])
                    # import pdb; pdb.set_trace()
                    # random.shuffle(labels)
                    # clean = clean[starts[0]:ends[0]]

                    starts = np.array([], dtype=np.int64)
                    ends = np.array([], dtype=np.int64)

                if target_sample_rate==8000:
                    starts = starts // 2
                    ends = ends // 2
                elif target_sample_rate==16000:
                    pass
                else:
                    raise ValueError(f"Unsupported sample rate: {target_sample_rate}")

                # load noise
                noise = random_load_audio_from_list(
                    self.noise_list,
                    sample_rate=target_sample_rate)

                # load room impulse response (RIR)
                rir = None
                if self.reverb_list:
                    rd_reverb = np.random.uniform(0,1)
                    if rd_reverb < revert_prob:
                        rir = random_load_audio_from_list(
                            self.reverb_list,
                            sample_rate=target_sample_rate)

                snr_dbs = self.params.data['snr_dbs']
                snr_db = snr_dbs[np.random.randint(0,len(snr_dbs))]

                # repeat or crop clean and noise to target length
                audio_sn, audio_s, starts, ends  = synthesize_audio_with_labels(
                    clean,
                    noise,
                    starts,
                    ends,
                    rir,
                    snr_db,
                    min_amp=self.params.data['min_amp'],
                    max_amp=self.params.data['max_amp'],
                    target_length=target_length,
                    sample_rate=target_sample_rate,
                    is_short_segments_remove=False,)
                if self.params.data.debug:
                    import sounddevice as sd
                    sd.play(audio_sn, samplerate=target_sample_rate)
                # vad = np.zeros_like(audio_sn, dtype=np.float32)
                # vad[start:end] = 1.0  # Mark the valid region
                
                # import matplotlib.pyplot as plt
                # plt.figure(figsize=(10, 4))
                # plt.subplot(3, 1, 1)
                # plt.plot(audio_sn, label='Clean Speech')
                # plt.plot(vad)
                # plt.title(f"Clean Speech Waveform - {wavname}")
                # plt.ylim(-1.1, 1.1)
                
                # plt.subplot(3, 1, 2)
                # plt.plot(audio_s, label='Clean Speech')
                # plt.plot(vad)
                # plt.title(f"Clean Speech Waveform - {wavname}")
    
                
                # plt.subplot(3, 1, 3)
                # plt.plot(audio_reverb, label='Noisy Speech')
                # plt.plot(vad)
                # plt.show()

                if is_dc_removal:
                    audio_sn = dc_remove(audio_sn)
                    audio_s = dc_remove(audio_s)
                # print(f"[Process {self.proc_pid}] {wavname} -> {snr_db}dB")
     
                if self.debug:
                    hop_size=self.params.train['feature']['hop_size']

                    self._debug_plot(
                        audio_sn, audio_s, snr_db,
                        label=(starts // hop_size, ends // hop_size))

                else:
                    tfrecord_name = re.sub(r'^wavs', path_tfrecord,
                        re.sub(r'\.(wav|flac)$', f'_{snr_db}-db_{self.info}_{k}_proc_id{self.proc_pid}.tfrecord', wavname))
                    create_raw_tfrecord(tfrecord_name, audio_sn, (starts, ends))
                    self.success_dict[self.proc_pid] += [tfrecord_name]

    def _debug_plot(
            self,
            audio_sn: np.array,
            audio_s: np.array,
            snr_db: float | int,
            vmin: float=-80.0,
            vmax: float=10.0,
            label: tuple=(10,100)) -> None:
        """
        Plot the audio signals and 
        their spectrograms for debugging.
        """

        feat_extractor = FeatureExtractor(
            params=self.params,
        )

        feat_sn, spec_sn, states_audio_sn = feat_extractor(
            tf.constant([audio_sn], dtype=tf.float32))
        feat_s, spec_s, states_audio_s = feat_extractor(
            tf.constant([audio_s], dtype=tf.float32))
        # import pdb; pdb.set_trace()
        vad = np.zeros(feat_sn.shape[1], dtype=np.float32)
        for ss, ee in zip(label[0], label[1]):
            vad[ss:ee] = 1
        
        # spec_sn = tf_stft([audio_sn], frame_size, hop_size, fft_size)
        # spec_s = tf_stft([audio_s], frame_size, hop_size, fft_size)
        logspec_sn = 20 * tf_log10_eps(tf.abs(spec_sn[0])).numpy()
        logspec_s = 20 * tf_log10_eps(tf.abs(spec_s[0])).numpy()
        logmel_sn = 10 * feat_sn[0].numpy()

        plot_spectrograms(
            images=[logspec_sn.T, logspec_s.T, logmel_sn.T],
            titles=[f"noisy logspec {snr_db}dB", "clean logspec", "noisy feat"],
            vmin_vmax=[(vmin, vmax), (vmin, vmax), (vmin, vmax)],
            show_colorbar=True,
            cmap="pink_r",  # or your preferred colormap
            show_fig=False   # set to False if you just want to save
            )
        plt.plot(vad * 22)
        plt.show()

def data(params: SKTaskParams) -> None:
    """Prepare tfrecords data for SE task

    Args:
        params (SKTaskParams): Task parameters

    """
    params_data = params.data
    force_download = params_data['force_download']

    if force_download:
        corpora = params_data['corpora']
        for d in corpora:
            corpus = d['name']
            type_cropus = d['type']
            corpus_download(corpus, type_cropus)

    speech_list = {'train': [], 'val': []}
    garb_list = {'train': [], 'val': []}
    noise_type2list = {}
    reverb_list = {'train': [], 'val': []}

    for corpus in params_data['corpora']:
        name = corpus['name']
        ctype = corpus['type']
        split = corpus['split']
        loader = SKDatasetFactory.get(name)
        files = loader(corpus)

        if ctype == 'speech':
            if split=="train":
                speech_list['train'].extend(files)
            elif split=="val":
                speech_list['val'].extend(files)
            elif split=="train-val":
                speech_list['train'].extend(files['train'])
                speech_list['val'].extend(files['val'])
            else:
                raise ValueError(f"Unknown split type: {split} for corpus {name}")
        elif ctype == 'garbage':
            if split == "train":
                garb_list['train'].extend(files)
            elif split == "val":
                garb_list['val'].extend(files)
            elif split == "train-val":
                garb_list['train'].extend(files['train'])
                garb_list['val'].extend(files['val'])
            else:
                raise ValueError(f"Unknown split type: {split} for corpus {name}")
        elif ctype == 'noise':
            if split == "train-val":
                noise_type2list[name] = files
            elif split == "train":
                noise_type2list[name] = {'train': files}
            elif split == "val":
                noise_type2list[name] = {'val': files}
            else:
                raise ValueError(f"Unknown split type: {split} for corpus {name}")
        elif ctype == 'reverb':
            if params_data['reverb_prob'] > 0:
                if split == "train":
                    reverb_list['train'].extend(files)
                elif split == "val":
                    reverb_list['val'].extend(files)
                elif split == "train-val":
                    reverb_list['train'].extend(files['train'])
                    reverb_list['val'].extend(files['val'])
                else:
                    raise ValueError(f"Unknown split type: {split} for corpus {name}")
        else:
            raise ValueError(f"Unknown corpus type: {ctype} for corpus {name}")

    sets = ['train','val']
    tot_success_dict = {'train': [], 'val': []}
    for train_set in sets:
        random.shuffle(speech_list[train_set])
        if params_data['num_samples_per_noise'][train_set] is None:
            # If num_samples_per_noise is None, use all available samples
            num_samples = len(speech_list[train_set])
        else:
            num_samples = np.minimum(
                len(speech_list[train_set]),
                params_data['num_samples_per_noise'][train_set])
        lst = np.array(speech_list[train_set][:num_samples], dtype=object)
        speech_list_split = np.array_split(
            lst,
            params_data['num_processes'])
        
        random.shuffle(garb_list[train_set])
        lst = np.array(garb_list[train_set], dtype=object)
        garb_list_split = np.array_split(
            lst,
            params_data['num_processes'])
        
        reverb_list_set = reverb_list[train_set]
        for noise_type in list(noise_type2list.keys()):
            log.info(f"Processing [{train_set}] set with [{noise_type}] noise")
            noise_list = noise_type2list[noise_type][train_set]
            manager = multiprocessing.Manager()
            success_dict = manager.dict({i: [] for i in range(params_data['num_processes'])})

            processes = []
            for i in range(params_data['num_processes']):
                audio_type_lists = {
                    'speech': speech_list_split[i],
                    'garbage': garb_list_split[i],
                    'noise': noise_list,
                    'reverb': reverb_list_set
                }
                proc = FeatMultiProcsClass(
                        i, # process id
                        audio_type_lists,
                        success_dict,
                        params,
                        info=noise_type,
                        debug=params.data['debug'])
                processes.append(proc)

            if params.data['debug']:
                for proc in processes:
                    proc.run()
            else:
                for proc in processes:
                    proc.start()

                for proc in processes:
                    proc.join()

            for lst in success_dict.values():
                tot_success_dict[train_set] += lst

    for train_set in sets:
        file_path = Path(params_data['path_tfrecord']) / params_data['tfrecord_datalist_name'][train_set]
        with file_path.open('w') as file:
            for tfrecord in tot_success_dict[train_set]:
                file.write(f'{Path(tfrecord).as_posix()}\n')
