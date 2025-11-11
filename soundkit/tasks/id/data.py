''' prepare tfrecords data for SE task '''
import os
import logging
import re
import multiprocessing
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from soundkit.utils.tf_basic_math import tf_log10_eps
from soundkit.utils.feature_utils import FeatureExtractor
from soundkit.defines import SKTaskParams
from soundkit.utils.basic_dsp import dc_remove
from soundkit.utils.download_api import corpus_download
from soundkit.utils.audio import audio_read, random_load_audio_from_list, synthesize_audio_with_labels
from soundkit.utils.plot_api import plot_spectrograms
from soundkit.datasets import SKDatasetFactory
from soundkit.utils.np_feature_utils import FeatureExtractor_np
from .datasets import create_tfrecord
from .utils.sort import grouping_spks_sentences

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
            snr_dbs: float | int = 100,
            debug: bool = False,
            ):
        super().__init__()

        self.proc_pid = proc_pid
        self.audio_type_lists = audio_type_list
        self.success_dict = success_dict
        self.params = params  # includes noise_files, snr_dbs, reverb_prob, output_dir, etc.

        self.debug= debug
        self.snr_dbs= snr_dbs

        params_feat = params.train.feature

        self.feat_inst  = FeatureExtractor_np(
                                feat_type       = params_feat.type,
                                frame_len       = params_feat.frame_size,
                                hop_len         = params_feat.hop_size,
                                fft_len         = params_feat.fft_size,
                                sampling_rate   = params.data.signal.sampling_rate,
                                mel_bins        = params_feat.bins,
                                stream          = False)

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
        target_length = int(target_length)
        # print(f"[Process {self.proc_pid}] Started with {len(self.speech_list)} files.")

        speech_list = self.audio_type_lists['speech']
        noise_types= self.audio_type_lists['noise_types']
        noise_type2list = self.audio_type_lists['noise_type2list']
        train_set = self.audio_type_lists['train_set']
        reverb_list = self.audio_type_lists['reverb'][train_set]

        for idx, sample in enumerate(tqdm(speech_list, desc=f"Processing {self.proc_pid}", unit="file", leave=False)):
            # load clean speech

            spk_id = sample[0]
            wavname, label = sample[1]

            clean = audio_read(
                wavname,
                sample_rate=target_sample_rate)

            starts = np.array([seg['start'] for seg in label])
            ends = np.array([seg['end'] for seg in label])
            
            for noise_type in noise_types:
                noise_list= noise_type2list[noise_type][train_set]
                for snr_db in self.snr_dbs:

                    id = np.random.randint(0, len(ends))
                    # id = np.argmax(ends - starts)
                    clean = clean[starts[id]:ends[id]]

                    if len(clean) > target_length:
                        s = np.random.randint(0, len(clean) - target_length + 1)
                        clean = clean[s:s+target_length]
                        starts = np.array([0])
                        ends = np.array([target_length-1])
                    else:
                        starts = np.array([0])
                        ends = np.array([len(clean)-1])
                        clean = np.pad(clean, [0,target_length-len(clean)])

                    if target_sample_rate==8000:
                        starts = starts // 2
                        ends = ends // 2
                    elif target_sample_rate==16000:
                        pass
                    else:
                        raise ValueError(f"Unsupported sample rate: {target_sample_rate}")


                    # load noise
                    noise = random_load_audio_from_list(
                        noise_list,
                        sample_rate=target_sample_rate)

                    # load room impulse response (RIR)
                    rir = None
                    if reverb_list:
                        rd_reverb = np.random.uniform(0,1)
                        if rd_reverb < revert_prob:
                            rir = random_load_audio_from_list(
                                reverb_list,
                                sample_rate=target_sample_rate)

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
                        sample_rate=target_sample_rate)

                    if is_dc_removal:
                        audio_sn = dc_remove(audio_sn)
                        audio_s = dc_remove(audio_s)
                    feat, spec = self.feat_inst(audio_sn)
                    hop_size=self.params.train['feature']['hop_size']
                    label=(starts // hop_size, ends // hop_size)

                    # print(f"[Process {self.proc_pid}] {wavname} -> {snr_db}dB")
                    dir = path_tfrecord + f'/spk-{spk_id}/{noise_type}/{snr_db}-db'
                    basename = re.sub(r'\.(wav|flac)$', '.tfrecord', os.path.basename(wavname))
                    tfrecord_name = os.path.join(dir, basename)
                    if self.debug:
                        self._debug_plot(
                            audio_sn, audio_s, snr_db,
                            label=label)
                    else:
                        os.makedirs(dir, exist_ok=True)
                        # create_raw_tfrecord(tfrecord_name, audio_sn, (starts, ends))
                        create_tfrecord(tfrecord_name, feat, label)
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

    def group_speakers(speech_list):
        "group speakers by their ID"
        grouped = defaultdict(list)

        for filepath, segments in speech_list:
            parts = filepath.split('/')
            speaker_id = parts[3]  # '200'
            grouped[speaker_id].append((filepath, segments))

        # Convert to list of lists
        spks = [[spk, files] for spk, files in grouped.items()]
        return spks

    def extract_sentences(spks,  sentences=10):
        """
        Extract a specified number of sentences from the speech list.
        """
        flattened = []
        for spk in spks:
            id, files = spk

            files = files[:sentences]  # Limit to the first 'sentences' files

            for file in files:
                flattened.append([id, file])  # Keep speaker ID and segments

        return flattened


    sets = ['train','val']
    ppls_per_group = params.data.ppls_per_group
    num_sentences = params.data.num_sentences

    spks = group_speakers(speech_list['train']) + group_speakers(speech_list['val'])
    len_tot= len(spks)
    spks = spks[:len_tot]  # Ensure the length is a multiple of 64
    len_tr = int(len_tot * 0.8)
    len_tr = ppls_per_group*(len_tr // ppls_per_group)

    len_te = len_tot-len_tr
    len_te = ppls_per_group*(len_te // ppls_per_group)

    speech_list['train'] = extract_sentences(spks[:len_tr], sentences=num_sentences)
    speech_list['val'] = extract_sentences(spks[len_tr:len_tr + len_te], sentences=num_sentences)

    tot_success_dict = {'train': [], 'val': []}
    for train_set in sets:
        lst = np.array(speech_list[train_set], dtype=object)
        speech_list_split = np.array_split(
            lst,
            params_data['num_processes'])

        # for noise_type in list(noise_type2list.keys()):
        #     for snr_db in params_data['snr_dbs']:
        log.info(f"Processing [{train_set}] set with ")
        # noise_list = noise_type2list[noise_type][train_set]
        manager = multiprocessing.Manager()
        success_dict = manager.dict({i: [] for i in range(params_data['num_processes'])})

        processes = []
        for i in range(params_data['num_processes']):
            audio_type_lists = {
                'speech': speech_list_split[i],
                'noise_types': list(noise_type2list.keys()),
                'noise_type2list': noise_type2list,
                'train_set': train_set,
                'reverb': reverb_list
            }
            proc = FeatMultiProcsClass(
                    i, # process id
                    audio_type_lists,
                    success_dict,
                    params,
                    snr_dbs=params_data['snr_dbs'],
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

        lst  = tot_success_dict[train_set]
        grouping_spks_sentences(lst, file_path)
        