''' prepare tfrecords data for SE task '''
import logging
from pathlib import Path
import random
import re
import multiprocessing
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from soundkit.utils.tf_basic_math import tf_log10_eps
from soundkit.defines import SKTaskParams
from soundkit.utils.basic_dsp import dc_remove
from soundkit.utils.download_api import corpus_download
from soundkit.utils.audio import audio_read, random_load_audio_from_list, synthesize_audio
from soundkit.utils.plot_api import plot_spectrograms
from soundkit.datasets import SKDatasetFactory
from soundkit.utils.feature_utils import FeatureExtractor
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
        from scipy.stats import truncnorm

        def get_snr_sample(batch_size=1):
            # Defining the bounds and shape
            lower, upper = -5, 30
            mu, sigma = 10, 10
            
            # Standardizing the bounds for scipy
            a, b = (lower - mu) / sigma, (upper - mu) / sigma
            
            # Sample from the distribution
            snr_samples = np.random.uniform(lower, upper, batch_size)  # to ensure randomness across processes
            # snr_samples = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=batch_size)
            return snr_samples

        target_sample_rate = self.params.data['signal']["sampling_rate"]
        is_dc_removal = self.params.data['signal']["dc_removal"]
        revert_prob = self.params.data["reverb_prob"]
        path_tfrecord=self.params.data['path_tfrecord']
        target_length = int(self.params.data['target_length_in_secs'] * target_sample_rate)
        # print(f"[Process {self.proc_pid}] Started with {len(self.speech_list)} files.")
        random.shuffle(self.noise_list)
        for id_clean, wavname in enumerate(tqdm(self.speech_list, desc=f"Processing {self.proc_pid}", unit="file", leave=False)):
            # load clean speech
            clean = audio_read(
                wavname,
                sample_rate=target_sample_rate)

            if len(clean) == 0:
                log.warning(f"Skipping empty audio file: {wavname}")
                continue

            # load noise
            noise = random_load_audio_from_list(
                self.noise_list,
                sample_rate=target_sample_rate,
                id = id_clean % len(self.noise_list)
            )

            # load room impulse response (RIR)
            rir = None
            if self.reverb_list:
                rd_reverb = np.random.uniform(0,1)
                if rd_reverb < revert_prob:
                    rir = random_load_audio_from_list(
                        self.reverb_list,
                        sample_rate=target_sample_rate)
            if 0:
                snr_dbs = self.params.data['snr_dbs']
                snr_db = snr_dbs[np.random.randint(0,len(snr_dbs))]
            else:
                snr_db = get_snr_sample(batch_size=1)[0]

            # repeat or crop clean and noise to target length
            audio_sn, audio_s, start, end  = synthesize_audio(
                clean,
                noise,
                rir,
                snr_db,
                min_amp=self.params.data['min_amp'],
                max_amp=self.params.data['max_amp'],
                target_length=np.minimum(target_length, len(clean)),
                sample_rate=target_sample_rate)
            
            # # add wind noise with 10% probability
            # pb = np.random.uniform(0,1)

            # if pb < 0.1:
                
            #     duration_secs =self.params.data['target_length_in_secs']
            #     wind = wind_noise(
            #         duration_secs,
            #         fs=target_sample_rate)  # intensity between 0 and 10 dB
            #     gain = np.random.uniform(0.1,1)
            #     audio_sn += gain * wind  # mix wind noise at a lower level
            #     amp = np.maximum(np.max(np.abs(audio_sn)), 1e-8)
            #     gain = np.random.uniform(0.03, 0.95) / amp
            #     audio_sn *= gain
            #     audio_s *= gain

            if is_dc_removal:
                audio_sn = dc_remove(audio_sn)
                audio_s = dc_remove(audio_s)
            # print(f"[Process {self.proc_pid}] {wavname} -> {snr_db}dB")
            
            snr_db = round(snr_db * 100)
            if self.debug:
                import sounddevice as sd
                sd.play(audio_sn, samplerate=target_sample_rate)
                self._debug_plot(audio_sn, audio_s, snr_db)
            else:
                tfrecord_name = re.sub(r'^wavs', path_tfrecord,
                    re.sub(r'\.(wav|flac)$', f'_{snr_db}_100-db_{self.info}.tfrecord', wavname))

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

        feat_extractor = FeatureExtractor(
            params=self.params,
        )

        feat_sn, spec_sn, states_audio_sn = feat_extractor(
            tf.constant([audio_sn], dtype=tf.float32))
        feat_s, spec_s, states_audio_s = feat_extractor(
            tf.constant([audio_s], dtype=tf.float32))

        
        logspec_sn = 10 * tf_log10_eps(tf.abs(spec_sn[0])**2).numpy()
        logspec_s = 10 * tf_log10_eps(tf.abs(spec_s[0])**2).numpy()
        
        if self.params.train['feature']['type'] in ('mel', 'logpspec', 'hybrid'):
            logmel_sn = 10 * feat_sn[0].numpy()
        elif self.params.train['feature']['type'] in ('pspec'):
            logmel_sn = 10 * tf_log10_eps(tf.abs(feat_sn[0])**2).numpy()
        elif self.params.train['feature']['type'] in ('pspec', 'spec', "erb_complex", "hybrid_mag", "erb_mag"):
            logmel_sn = 20 * tf_log10_eps(tf.abs(feat_sn[0])**2).numpy()
        plot_spectrograms(
            images=[logspec_sn.T, logspec_s.T, logmel_sn.T],
            titles=[f"noisy logspec {snr_db}dB", "clean logspec", "noisy feat"],
            vmin_vmax=[(vmin, vmax), (vmin, vmax), (vmin, vmax)],
            show_colorbar=True,
            cmap="pink_r",  # or your preferred colormap
            show_fig=True   # set to False if you just want to save
            )

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
    for key in speech_list.keys():
        lst_ch = []
        lst_else = []
        for v in speech_list[key]:
            if re.search(r"(thchs30|mandarin|MAGICDATA)", v):
                lst_ch.append(v)
            else:
                lst_else.append(v)
        # random.shuffle(lst_ch)
        # random.shuffle(lst_en)

        # if key == "train":
        #     len0 = min(len(lst_ch), len(lst_en))
        #     lst = lst_ch[:len0] + lst_en[:len0]
        # else:
        #     lst = lst_ch + lst_en
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        speech_list[key] = lst_ch + lst_else

    sets = ["train", "val"]
    tot_success_dict = {'train': [], 'val': []}
    snr_dbs_default = params_data['snr_dbs']

    for train_set in sets:
        random.shuffle(speech_list[train_set])
        if params_data['num_samples_per_noise'][train_set] is None:
            # If num_samples_per_noise is None, use all available samples
            num_samples = len(speech_list[train_set])
        else:
            num_samples = np.minimum(
                len(speech_list[train_set]),
                params_data['num_samples_per_noise'][train_set])

        # Robust split for Python lists of possibly ragged elements
        # Avoid np.array_split which tries to coerce to ndarray and can fail on heterogeneous sequences
        split_points = np.linspace(
            0,
            num_samples,
            params_data['num_processes'] + 1,
            dtype=int
        )
        speech_list_split = [
            speech_list[train_set][split_points[i]:split_points[i + 1]]
            for i in range(params_data['num_processes'])
        ]
        reverb_list_set = reverb_list[train_set]
        for noise_type in list(noise_type2list.keys()):
            log.info(f"Processing [{train_set}] set with [{noise_type}] noise")
            
            # if noise_type=="wind_noise":
            #     # for wind noise, use only specific SNRs
            #     params.data['snr_dbs'] = [-15, -12, -9, -6, -3, 0]
            # else:
            #     params.data['snr_dbs'] = snr_dbs_default
            
            params.data['snr_dbs'] = snr_dbs_default
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
