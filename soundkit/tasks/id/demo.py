"""Demo script for deploying TFLite model to neuralSPOT EVB or running on PC."""
import os
import logging
import subprocess
import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
from soundkit.defines import SKTaskParams
from soundkit.utils.feature_utils import FeatureExtractor
from soundkit.utils.np_feature_utils import FeatureExtractor_np
from soundkit.utils.pyaudio_animation_id import AudioShowClass
from soundkit.utils.calculate_feat_stats import load_feat_stats
from soundkit.utils.TFLiteAudioModel import TFLiteAudioModel
from soundkit.utils.generate_feature_c_files import generate_feature_c_files
from soundkit.utils.basic_dsp import DCRemover
from soundkit.utils.converter_fix_point import (
        fakefix_tf,
        int2str_array
    )
from soundkit.utils.tf_stft import gen_stft_win
from soundkit.utils.mel import gen_mel_c
from .export import export

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

def demo(params: SKTaskParams):
    """
    Deploy a TFLite model to neuralSPOT and install dependencies.

    Args:
        params (SKTaskParams): Task parameters
    """
    # === export TFLite File ===
    tflite_filename_src = f"{params.name}_{params.demo['dtype']}.tflite"
    tflite_path_src = Path(params.demo['tflite_dir']) / tflite_filename_src
    log.info(f"🧪 Exporting TFLite model from {tflite_path_src}")
    params.export['epoch_loaded'] = params.demo['epoch_loaded']
    params.export['tflite_dir'] = params.demo['tflite_dir']
    params.export["num_frames_infer"] = params.demo["num_frames_infer"]
    params.export["calibration_samples"] = params.demo["calibration_samples"]
    params.export["dtype"] = params.demo["dtype"]
    export(params)
    if params.demo.platform == 'evb':
        demo_evb(params, tflite_path_src)
    elif params.demo.platform == 'pc':
        demo_pc(params, tflite_path_src)
    else:
        raise ValueError(
            f"Unsupported platform: {params.demo.platform}. ",
             "Supported platforms are 'evb' and 'pc'.")

def demo_evb(
        params: SKTaskParams,
        tflite_path_src: str):
    """
    Deploy a TFLite model to neuralSPOT and install dependencies.

    Args:
        params (SKTaskParams): Task parameters
        tflite_path_src: str
    """
    # === Setup Variables ===

    current_dir = Path.cwd().resolve()
    log.info(f"🔧 Current working directory: {current_dir}")

    tflite_filename = "net.tflite"

    tflm_version = "ns_tflm_v1_0_0"

    evb_src_tflm_dir = Path(params.demo['evb_dir']) / 'src' # "tflm"

    # === Download neuralSPOT ===
    repo_url = "https://github.com/AmbiqAI/neuralSPOT.git"
    neuralSPOT = "neuralSPOT_autodeploy"
    neuralspot_path = Path(f"../{neuralSPOT}").resolve()
    if not os.path.exists(neuralspot_path):
        subprocess.run(["git", "clone", repo_url, neuralspot_path], check=True)
        log.info(f"📦 Cloned {neuralSPOT} to {neuralspot_path}")
    else:
        log.info(f"✅ {neuralSPOT} already exists at {neuralspot_path}")
    # === Checkout specific commit ===
    subprocess.run(["git", "checkout", "14c29b246"], cwd=neuralspot_path, check=True)
    log.info(f"🔄 Checked out neuralSPOT commit 14c29b246")
    # === Generate Feature C Files ===
    log.info("🧪 Generating feature C files")

    checkpoint_dir = f"{params.train['path']['checkpoint_dir']}"

    # === Generate C Code STFT Window ===

    # Extract parameters for readability
    feat_params = params.train['feature']
    framesize = feat_params['frame_size']
    hopsize = feat_params['hop_size']

    stft_win_name='stft_win_coeff'
    win_coeff = gen_stft_win(
        win_size=framesize,
        hop=hopsize)
    win_coeff = fakefix_tf(win_coeff, 16, 15)

    # Build the file content as a list of strings
    lines = [
        '#include <stdint.h>\n',
        f"// stft window_coeff (framesize={framesize}, hopsize={hopsize})",
        int2str_array(stft_win_name, win_coeff.numpy() * 32768, nbits=16)
    ]

    c_code = "\n".join(lines)
    Path(f"{evb_src_tflm_dir}/{stft_win_name}.c").write_text(c_code)

    # === Generate C Code Filter Banks ===


    filterbank_name='filter_banks'

    feat_extractor = FeatureExtractor(
        params=params,
    )

    if feat_extractor.mel_filter is None:
        fbanks=None
    else:
        fbanks = tf.identity(feat_extractor.mel_filter)
        fbanks = fakefix_tf(fbanks, 16, 15).numpy().T
    gen_mel_c(
        f"{evb_src_tflm_dir}/{filterbank_name}.c",
        filterbank_name,
        mel_filters=fbanks,
        bank_type=params.train['feature']['type'])

    # === Generate feature statstics ===

    stats_name = 'stats.pkl'
    stats = load_feat_stats( \
        dir=checkpoint_dir, \
        stats_name=stats_name)

    generate_feature_c_files(
        file_name=params.demo.filename,
        param_struct_name=params.demo.param_struct_name,
        dir=evb_src_tflm_dir,
        feature_mean=stats['nMean_feat'],
        feature_std=stats['nInvStd'],
        sampling_rate=params.data['signal']['sampling_rate'],
        fftsize=feat_params['fft_size'],
        winsize_stft=feat_params['frame_size'],
        hopsize_stft=feat_params['hop_size'],
        num_mfltrBank=feat_extractor.dim_feat,
        is_dcrm=int(params.data['signal']['dc_removal']),
        pre_gain_q1=params.demo['pre_gain'],
        lookahead=params.train['num_lookahead'],
        stft_win_coeff_name=stft_win_name,
        filterbank_name=filterbank_name,
        task=params.project,
        num_frames_infer=params.demo["num_frames_infer"],
        feature_type=params.train['feature']['type'],
    )

    # === Define Key Paths ===
    tools_dir = Path(f"../{neuralSPOT}/tools").resolve()
    tflite_path_dst = tools_dir / tflite_filename
    neuralspot_root = Path(f"../{neuralSPOT}").resolve()

    # === Copy TFLite File to neuralSPOT/tools ===
    log.info(f"📦 Copying TFLite to {tflite_path_dst}")
    tools_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(tflite_path_src, tflite_path_dst)

    # === Setup venv and install ===
    log.info(f"🔧 Setting up virtual environment at {neuralspot_root}")
    os.chdir(neuralspot_root)
    (neuralspot_root / "projects/autodeploy").mkdir(parents=True, exist_ok=True)
    # Ensure uv is on PATH for subprocess calls
    os.environ["PATH"] = f"{Path.home()}/.local/bin:" + os.environ.get("PATH", "")

    # Try to locate or download helios-aot
    aot_candidates = [
        Path.home() / "Documents" / "helios-aot",
        Path.home() / "helios-aot",
    ]
    aot_path = None
    for p in aot_candidates:
        if p.exists():
            aot_path = p
            break
    if aot_path is None:
        try:
            log.info("📥 Downloading helios-aot to ~/helios-aot")
            subprocess.run([
                "git", "clone", "https://github.com/AmbiqAI/helios-aot.git",
                str(Path.home() / "helios-aot")
            ], check=True)
            aot_path = Path.home() / "helios-aot"
        except subprocess.CalledProcessError:
            log.warning("helios-aot not found and clone failed; proceeding without explicit AOT add. If uv sync fails, set HELIOS AOT path manually.")

    # Pin Python and sync root env for neuralSPOT
    subprocess.run(["uv", "python", "pin", "3.12.11"], cwd=neuralspot_root, check=True)

    # If we have helios-aot, add it to the tools project and sync there too
    tools_project_dir = neuralspot_root / "tools"
    if aot_path and tools_project_dir.exists():
        try:
            # Remove any previous entry and add our local path
            subprocess.run(["uv", "remove", "helios-aot"], cwd=tools_project_dir, check=False)
            subprocess.run(["uv", "add", str(aot_path)], cwd=tools_project_dir, check=True)
            subprocess.run(["uv", "sync"], cwd=tools_project_dir, check=True)
        except subprocess.CalledProcessError as e:
            log.warning(f"Failed to add/sync helios-aot: {e}")

    # Also sync at neuralSPOT root to ensure its venv resolves dependencies
    try:
        subprocess.run(["uv", "sync"], cwd=neuralspot_root, check=True)
    except subprocess.CalledProcessError as e:
        log.error(
            "uv sync failed in neuralSPOT root.\n"
            "- Ensure 'uv' is installed and on PATH (~/.local/bin).\n"
            f"- If a local dependency like helios-aot is required, verify it exists at: {aot_path or '[not found]'}\n"
            "  or configure the tools project to point to your actual path."
        )
        raise

    # === Ubuntu Fix: Ensure SVD path exists ===
    log.info("🐧 Fixing SVD path for Ubuntu")
    svd_dir = neuralspot_root / "extern/AmbiqSuite/R5.3.0/pack/svd"
    svd_dir.mkdir(parents=True, exist_ok=True)

    svd_src = neuralspot_root / "extern/AmbiqSuite/R5.3.0/pack/SVD/apollo510.svd"
    svd_dst = svd_dir / "apollo510.svd"

    if not os.path.exists(svd_dst):
        shutil.copy(svd_src, svd_dst)
        log.info(f"✅ SVD file copied to {svd_dst}")
    else:
        log.info(f"✅ SVD file already exists at {svd_dst}")

    # === Run ns_autodeploy ===
    log.info("⚙️  Running ns_autodeploy")
    os.chdir(tools_dir)
    subprocess.run([
        "../.venv/bin/ns_autodeploy",
        "--tflite-filename", f"./{tflite_filename}",
        "--tensorflow-version", tflm_version,
        "--arena-size-scratch-buffer-padding", "10",
    ], check=True)

    # === Copy output files ===
    log.info("📤 Copying validator output files:")
    validator_src = neuralspot_root / "projects/autodeploy" / Path(tflite_filename).stem / "tflm_validator" / "src"
    target_dst_dir = Path(current_dir) / evb_src_tflm_dir
    target_dst_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "mut_model_data.h",
        "mut_model_init.cc",
        "mut_model_metadata.h",
        "tflm_ns_model.h"
    ]

    for fname in files_to_copy:
        src_file = validator_src / fname
        dst_file = target_dst_dir / fname
        shutil.copy(src_file, dst_file)
        log.info(f"  - {fname} copied to {dst_file}")

    # === Build and Deploy ===
    log.info("⚙️  Building and deploying to neuralSPOT")

    os.chdir(current_dir)
    os.chdir(params.demo['evb_dir'])

    subprocess.run(["make", "clean"], check=True)

    subprocess.run(["make"], check=True)
    subprocess.run(["make", "deploy"], check=True)

    subprocess.run(["make", "view"], check=True)

    log.info("✅ TFLite deployment and file transfer complete.")

def demo_pc(
        params_id: SKTaskParams,
        tflite_path_src: str):
    """Export ID task model with given parameters.

    Args:
        params (SKTaskParams): Task parameters
    """
    from ...cli import parse_config

    params_vad = parse_config("zoo/vad/freq_model/vad.yaml")

    params_list={
        'id': params_id,
        'vad': params_vad
    }

    def init_model_artifacts(params: SKTaskParams):

        if params.train.feature.type=='hybrid':
            mel_bins = params.train.feature.n_mels
        else:
            mel_bins = params.train.feature.bins
        feat_extractor = FeatureExtractor_np(
            feat_type=params.train.feature.type,
            frame_len=params.train.feature.frame_size,
            hop_len=params.train.feature.hop_size,
            fft_len=params.train.feature.fft_size,
            sampling_rate=params.data.signal.sampling_rate,
            mel_bins=mel_bins,
        )

        tflite_filename_src = f"{params.name}_{params.demo['dtype']}.tflite"
        tflite_path_src = Path(params.demo['tflite_dir']) / tflite_filename_src
        log.info(f"🧪 Exporting TFLite model from {tflite_path_src}")
        params.export['epoch_loaded'] = params.demo['epoch_loaded']
        params.export['tflite_dir'] = params.demo['tflite_dir']
        params.export["calibration_samples"] = params.demo["calibration_samples"]
        params.export["dtype"] = params.demo["dtype"]
        export(params)

        if params.train.standardization:
            stats = load_feat_stats(
                params.train['path']['checkpoint_dir'],
                'stats.pkl')
        else:
            stats = None

        return feat_extractor, stats, str(tflite_path_src)

    class IDModel:
        """VAD model class in tflite for processing audio input and returning VAD output."""
        def __init__(
                self,
                params_list: dict):
            self.params_list = params_list
            self.frames_vad_trigger_id = params_id.demo.frames_vad_trigger_id
            self.buffer = np.zeros(160 * self.frames_vad_trigger_id, dtype=np.float32)

            self.counts_registered = 0
            self.buffer_registered = np.zeros(
                [params_id.demo.num_utterances_registered, 64], dtype=np.float32)
            self.num_utterances_registered = params_id.demo.num_utterances_registered
            self.reg_d_vectors = {}
            self.dc_removers = {
                'vad': DCRemover()
            }

            if params_id.data.signal.dc_removal:
                self.dc_removers['id'] = DCRemover()

            self.vad_init()
            self.vad_reset()


        def vad_init(self):
            """Initialize the model state."""
            self.tflite_path_srcs = {}
            self.stats_list = {}
            self.feat_extractors = {}

            for key, params in params_list.items():
                feat_extractor, stats, tflite_path_src = init_model_artifacts(params)
                self.feat_extractors[key] = feat_extractor
                self.stats_list[key] = stats
                self.tflite_path_srcs[key] = tflite_path_src

        def vad_reset(self):
            """Reset the model state."""
            self.vad_counts=0
            self.buffer *= 0
            self.dc_removers['vad'].reset()

            if params_id.data.signal.dc_removal:
                self.dc_removers['id'].reset()

            self.models_tflite = {}
            for key, tflite_path_src in self.tflite_path_srcs.items():
                self.models_tflite[key] = TFLiteAudioModel(
                    interpreter_path=tflite_path_src,
                    dtype=params_list[key].demo.dtype,)

        def __call__(self,
                     inputs: np.ndarray, # input from microphone
                     register: str = None, # not used
                    ) -> np.ndarray: # output to AudioShowClass
            """Process input audio signal and return VAD output."""

            if not register['is_register']:
                self.counts_registered = 0

            feat_extractor = self.feat_extractors['vad']
            model_tflite = self.models_tflite['vad']
            stats = self.stats_list['vad']

            shape=inputs.shape

            inputs=inputs.flatten()

            inputs = self.dc_removers['vad'].process(inputs)

            features,_ = feat_extractor(inputs)

            if stats is not None:
                features = (features - stats['nMean_feat']) * stats['nInvStd']

            # input to the tflite model

            features = features.reshape((1, 1, -1)) # reshape to (batch_size, time_steps, dim_feat)
            outputs = model_tflite(features)

            outputs = outputs.flatten()

            request = {
                'is_register': register['is_register'],
                'counts': self.counts_registered,
                'info': {},
            }

            if outputs[0] < outputs[1]:
                outputs = np.ones(160, dtype=np.float32)*0.95
                self.buffer[self.vad_counts*160:(self.vad_counts+1)*160] = inputs
                self.vad_counts += 1

                if self.vad_counts == self.frames_vad_trigger_id:

                    # id processing
                    request = self.id_process(
                        self.buffer,
                        register=register,
                        request=request)

                    self.vad_reset()  # Reset after playing the audio

            else:
                outputs = np.zeros(160, dtype=np.float32)
                self.vad_counts = 0
            outputs = outputs.reshape(shape)

            return outputs, request

        def normalize(self, inputs: np.ndarray, eps: float = 10**-5) -> np.ndarray:
            """Normalize the input audio signal."""
            norm = np.maximum(np.sqrt(np.sum(inputs**2)), eps)
            return inputs / norm

        def id_process(
                self,
                inputs: np.ndarray, # input from microphone
                register: dict = {},
                request: dict = {},
            ) -> np.ndarray:

            """Process speaker identification NN inference."""
            feat_extractor = self.feat_extractors['id']
            model_tflite = self.models_tflite['id']
            stats = self.stats_list['id']

            # speaker identification inference
            for f in range(self.frames_vad_trigger_id):
                pcm = inputs[f*160:(f+1)*160]
                if params_id.data.signal.dc_removal:
                    pcm = self.dc_removers['id'].process(pcm)
                features,_ = feat_extractor(pcm)

                if stats is not None:
                    features = (features - stats['nMean_feat']) * stats['nInvStd']

                # input to the tflite model
                features = features.reshape((1, 1, -1)) # reshape to (batch_size, time_steps, dim_feat)

                outputs = model_tflite(features)

            d_vector = outputs.flatten()

            if register['is_register']: # registering mode
   
                self.buffer_registered[self.counts_registered] = self.normalize(d_vector)
                self.counts_registered += 1
                request['counts'] = self.counts_registered
                if self.counts_registered == self.num_utterances_registered:

                    request['is_register']=False

                    print(f"Registered {register['name']} with {self.counts_registered} samples.")

                    self.reg_d_vectors[register['name']] = np.mean(self.buffer_registered, axis=0)

                    self.counts_registered = 0

            else: # identification mode
                info = {}
                for name, vec in self.reg_d_vectors.items():

                    norm1 = np.sqrt(np.sum(vec**2)+10**-5)
                    norm2 = np.sqrt(np.sum(d_vector**2)+10**-5)

                    percent = 100 * np.sum(vec * d_vector) / (norm1 * norm2)
                    info[name] = round(percent, 2)
                request['info'] = info
            return request

    # Initialize the ID model with parameters
    id_model = IDModel(
        params_list=params_list
    )
    aud_handle = AudioShowClass(
        record_seconds=15,
        non_stop=True,
        proc_st=id_model
    )
