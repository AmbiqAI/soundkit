''' prepare tfrecords data for SE task '''
import os
import random
import re
import multiprocessing
from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ...utils.tf_stft import tf_stft
from ...utils.tf_basic_math import tf_log10_eps

from .datasets import create_raw_tfrecord
from ...defines import SKTaskParams
from ...utils.basic_dsp import dc_remove
from ...utils.download_api import corpus_download
from ...utils.audio import audio_read, random_load_audio_from_list, synthesize_audio
from ...utils.plot_api import plot_spectrograms
from ...datasets.register_datasets import DatasetRegistry


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
        target_length = self.params.data['target_length_in_secs'] * target_sample_rate
        # print(f"[Process {self.proc_pid}] Started with {len(self.speech_list)} files.")

        for wavname in tqdm(self.speech_list, desc=f"Processing {self.proc_pid}", unit="file", leave=False):
            # load clean speech

            clean = audio_read(
                wavname,
                sample_rate=target_sample_rate)

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
            audio_sn, audio_s, start, end  = synthesize_audio(
                clean,
                noise,
                rir,
                snr_db,
                min_amp=self.params.data['min_amp'],
                max_amp=self.params.data['max_amp'],
                target_length=target_length,
                sample_rate=target_sample_rate)

            if is_dc_removal:
                audio_sn = dc_remove(audio_sn)
                audio_s = dc_remove(audio_s)
            # print(f"[Process {self.proc_pid}] {wavname} -> {snr_db}dB")
            if self.debug:
                self._debug_plot(audio_sn, audio_s, snr_db)
            else:
                tfrecord_name = re.sub(r'^wavs', path_tfrecord,
                    re.sub(r'\.(wav|flac)$', f'_{snr_db}-db_{self.info}.tfrecord', wavname))

                create_raw_tfrecord(tfrecord_name, audio_sn, audio_s)
                self.success_dict[self.proc_pid] += [tfrecord_name]

    def _debug_plot(
            self,
            audio_sn: np.array,
            audio_s: np.array,
            snr_db: float | int,
            vmin: float=-80.0,
            vmax: float=10.0) -> None:
        """
        Plot the audio signals and 
        their spectrograms for debugging.
        """
        
        from ...utils.feature_utils import FeatureExtractor
        feat_extractor = FeatureExtractor(
            params=self.params,
        )
        
        frame_size = self.params.train['feature']['frame_size']
        hop_size = self.params.train['feature']['hop_size']
        fft_size = self.params.train['feature']['fft_size']
        feat_sn, spec_sn, states_audio_sn = feat_extractor(
            tf.constant([audio_sn], dtype=tf.float32))
        feat_s, spec_s, states_audio_s = feat_extractor(
            tf.constant([audio_s], dtype=tf.float32))
        
        
        # spec_sn = tf_stft([audio_sn], frame_size, hop_size, fft_size)
        # spec_s = tf_stft([audio_s], frame_size, hop_size, fft_size)
        logspec_sn = 20 * tf_log10_eps(tf.abs(spec_sn[0])).numpy()
        logspec_s = 20 * tf_log10_eps(tf.abs(spec_s[0])).numpy()
        logmel_sn = 20 * feat_sn[0].numpy()
        
  
        plot_spectrograms(
            images=[logspec_sn.T, logspec_s.T, logmel_sn.T],
            titles=[f"noisy logspec {snr_db}dB", "clean logspec", "noisy feat"],
            vmin_vmax=[(vmin, vmax), (vmin, vmax), (vmin, vmax)],
            show_colorbar=True,
            cmap="pink_r",  # or your preferred colormap
            show_fig=True   # set to False if you just want to save
            )

# def get_wavefiles(path_folder):
#     """Fetch all of noise files"""

#     lst = []
#     for root, _, files in os.walk(f'{path_folder}'):
#         for file in files:
#             if re.search(r'(wav$|flac$)', file):
#                 lst += [os.path.join(root, file.strip())]
#     return lst

# def get_noise_file_list(
#         ntype: str,) -> dict[str, list[str]]:
#     """
#     Get the list of noise file paths for a given noise type and set.

#     Args:
#         ntype (str): Noise type (e.g., 'musan', 'FSD50K', etc.)
#         set_name (str): 'train' or 'val'

#     Returns:
#         List[str]: File paths for noise
#     """
#     set_names = ['train', 'val']
#     files = {}
#     match ntype:

#         case 'wham_noise':
#             for set_name in set_names:
#                 folder = 'tr' if set_name == 'train' else 'cv'
#                 files[set_name] = get_wavefiles(f'wavs/noise/wham_noise/{folder}')

#             return files

#         case 'FSD50K' | 'ESC-50-master':
#             with open(f'wavs/noise/{ntype}/non_speech.csv', 'r') as f:
#                 lines = f.readlines()
#             lines = [line.strip() for line in lines]  # Skip header
#             random.shuffle(lines)
#             split = len(lines) // 5
#             files['train'] = lines[split:]
#             files['val'] = lines[:split]

#             return files

#         case 'musan':
#             music = get_wavefiles('wavs/noise/musan/music')
#             noise = get_wavefiles('wavs/noise/musan/noise')
#             lines = music + noise
#             random.shuffle(lines)
#             split = len(lines) // 5
#             files['train'] = lines[split:]
#             files['val'] = lines[:split]

#             return files

#         case 'RIRS_NOISES' | 'rirs_noises':
#             all_files = get_wavefiles(f'wavs/noise/RIRS_NOISES')
#             split = len(all_files) // 5
#             random.shuffle(all_files)
#             files['train'] = all_files[split:]
#             files['val'] = all_files[:split]
#             return files

#         case _:
#             raise ValueError(f"Unknown noise type: {ntype}")

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
    noise_type2list = {}
    reverb_list = {'train': [], 'val': []}

    for corpus in params_data['corpora']:
        name = corpus['name']
        ctype = corpus['type']
        split = corpus['split']
        loader = DatasetRegistry.get(name)
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
        elif ctype == 'noise':
            if split == "train-val":
                noise_type2list[name] = files
            elif split == "train":
                noise_type2list[name] = {'train': files}
            elif split == "val":
                noise_type2list[name] = {'val': files}
            else:
                raise ValueError(f"Unknown split type: {split} for corpus {name}")
        elif ctype == 'reverb' and params_data['reverb_prob'] > 0:
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
        num_samples = np.minimum(
            len(speech_list[train_set]),
            params_data['num_samples_per_noise'][train_set])
        speech_list_split = np.array_split(
            speech_list[train_set][:num_samples],
            params_data['num_processes'])
        reverb_list_set = reverb_list[train_set]
        for noise_type in list(noise_type2list.keys()):
            print(f"Processing [{train_set}] set with [{noise_type}] noise")
            noise_list = noise_type2list[noise_type][train_set]
            manager = multiprocessing.Manager()
            success_dict = manager.dict({i: [] for i in range(params_data['num_processes'])})

            processes = []
            for i in range(params_data['num_processes']):
                audio_type_lists = {
                    'speech': speech_list_split[i],
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
