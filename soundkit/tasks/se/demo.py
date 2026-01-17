""" DEMO for SE task"""
import os
import logging
import subprocess
import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
from soundkit.defines import SKTaskParams
from soundkit.utils.np_feature_utils import FeatureExtractor_np
from soundkit.utils.pyaudio_animation import AudioShowClass
from soundkit.utils.calculate_feat_stats import load_feat_stats
from soundkit.utils.TFLiteAudioModel import TFLiteAudioModel
from soundkit.utils.generate_feature_c_files import generate_feature_c_files
from soundkit.utils.basic_dsp import DCRemover
from soundkit.utils.np_stft import StreamingISTFT
from soundkit.utils.converter_fix_point import (
        fakefix_tf,
        int2str_array
    )
from soundkit.utils.tf_stft import gen_stft_win
from soundkit.utils.feature_utils import FeatureExtractor
from soundkit.utils.mel import gen_mel_c
from soundkit.utils.erb import ERB
from .export import export

erb = ERB(
    erb_subband_1=65,
    erb_subband_2=64,
    platform="numpy")

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
    params.export["calibration_samples"] = params.demo["calibration_samples"]
    params.export["num_frames_infer"] = params.demo["num_frames_infer"]
    params.export["dtype"] = params.demo["dtype"]
    export(params)

    # === Choose platform ===
    if params.demo.platform == 'evb':
        demo_evb(params, tflite_path_src)
    elif params.demo.platform == 'pc':
        demo_pc(params, tflite_path_src)
    else:
        raise ValueError(
            f"Unsupported platform: {params.demo.platform}. "
            "Supported platforms are 'evb' and 'pc'."
        )

def demo_evb(
        params: SKTaskParams,
        tflite_path_src: str
        ):
    """
    Deploy a TFLite model to neuralSPOT and install dependencies.

    Args:
        params (SKTaskParams): Task parameters
    """
    # === Setup Variables ===

    current_dir = Path.cwd().resolve()
    log.info(f"🔧 Current working directory: {current_dir}")


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
    fbanks = tf.identity(feat_extractor.mel_filter)
    fbanks = fakefix_tf(fbanks, 16, 15).numpy().T
    gen_mel_c(
        f"{evb_src_tflm_dir}/{filterbank_name}.c",
        filterbank_name,
        mel_filters=fbanks,
        bank_type=params.train['feature']['type'])

    if params.train['feature']['type'] in ['erb_complex', 'erb_mag']:
        invfilterbank_name='filter_banks_inv'
        fbanks = tf.identity(feat_extractor.mel_filter_inv)
        fbanks = fakefix_tf(fbanks, 16, 15).numpy().T

        gen_mel_c(
            f"{evb_src_tflm_dir}/{invfilterbank_name}.c",
            invfilterbank_name,
            mel_filters=fbanks,
            bank_type=params.train['feature']['type'])

    # === Generate feature statstics ===
    if params.train.standardization:
        stats_name = 'stats.pkl'
        stats = load_feat_stats(
            dir=checkpoint_dir,
            stats_name=stats_name)
    else:
        stats = None

    mean_t = stats['nMean_feat'] if stats is not None else None
    stdinv_t = stats['nInvStd'] if stats is not None else None

    generate_feature_c_files(
        file_name="def_nn3_se",
        param_struct_name="params_nn3_se",
        dir=evb_src_tflm_dir,
        feature_mean=mean_t,
        feature_std=stdinv_t,
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
        num_frames_infer=params.demo["num_frames_infer"],
        feature_type=params.train['feature']['type'],
    )

    # === Define Key Paths ===
    tflite_filename = "net.tflite"

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
        "--tensorflow-version", tflm_version
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
        params: SKTaskParams,
        tflite_path_src: str
        ):
    """Export se task model with given parameters.

    Args:
        params (SKTaskParams): Task parameters
    """
    checkpoint_dir = f"{params.train['path']['checkpoint_dir']}"

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
        platform="numpy"
    )
    hop_size = params.train.feature.hop_size

    interpreter = tf.lite.Interpreter(
        model_path=str(tflite_path_src)
    )
    interpreter.allocate_tensors()  # Needed before execution!

    if params.train.standardization:
        stats = load_feat_stats(checkpoint_dir, 'stats.pkl')
    else:
        stats = None

    model_tflite = TFLiteAudioModel(
        interpreter=interpreter,
        dtype=params.demo['dtype'],
    )

    class SEModel:
        """VAD model class in tflite for processing audio input and returning VAD output."""
        def __init__(
                self,
                model_tflite: TFLiteAudioModel,
                feat_extractor: FeatureExtractor_np,
                stats: dict | None = None,
                num_lookahead: int = 0):
            self.num_lookahead = num_lookahead
            self.model_tflite = model_tflite
            self.stats = stats
            self.feat_extractor = feat_extractor
            self.specs = []
            self.istft = StreamingISTFT(
                frame_len=params.train.feature.frame_size,
                hop_len=params.train.feature.hop_size,
                fft_len=params.train.feature.fft_size,
            )

            if params.train.feature.type == 'erb_complex':
                self.erb = erb

            if num_lookahead > 0:
                _, z_spec = feat_extractor(
                    np.zeros(hop_size, dtype=np.float32))  # Warm up the feature extractor
                for i in range(num_lookahead):
                    self.specs.append(z_spec.copy())
            if params.data['signal']['dc_removal']:
                self.dc_remover = DCRemover()

        def __call__(self,
                     inputs: np.ndarray # input from microphone
                    ) -> np.ndarray: # output to AudioShowClass
            """Process input audio signal and return VAD output."""
            inputs=inputs.flatten()
            if params.data['signal']['dc_removal']:
                inputs = self.dc_remover.process(inputs)
            features,spec_update = self.feat_extractor(inputs)
            self.specs.append(spec_update)
            spec = self.specs.pop(0)
            if self.stats is not None:
                features = (features - self.stats['nMean_feat']) * self.stats['nInvStd']

            # input to the tflite model
            features = features.reshape((1, 1, -1)) # reshape to (batch_size, time_steps, dim_feat)

            # reshape to (batch_size, time_steps, dim_feat, 2) for complex input
            if np.iscomplexobj(features):
                features = np.stack(
                    (features.real, features.imag), axis=-1)

            features =fakefix_tf(features, 32, 21).numpy()

            tfmask = self.model_tflite(features)

            pcm_out = self.post_procsessing(tfmask, spec)

            return pcm_out.reshape((-1,1))

        def post_procsessing(
                self,
                tfmask: np.ndarray,
                spec: np.ndarray) -> np.ndarray:
            """
            post processing after getting mask from tflite model
            Args:
                tfmask (np.ndarray): output mask from tflite model
                spec (np.ndarray): input complex spectrogram
            Returns:
                np.ndarray: time-domain waveform after ISTFT
            """

            if params.train.feature.type == 'erb_complex':
                tfmask = np.transpose(tfmask, axes=[0, 3, 1, 2]) # (B,T,F_erb,2) -> (B,2, T, F_erb)
                tfmask = erb.bs(tfmask)
                tfmask = np.transpose(tfmask, axes=[0, 2, 3, 1])  # (B,2, T, F_erb) -> (B,T,F_erb,2)
                tfmask = tfmask[:, :, :, 0] + 1j * tfmask[:, :, :, 1]
            elif params.train.feature.type == 'erb_mag':
                tfmask = erb.bs(tfmask[..., 0])

            tfmask = tfmask.flatten()

            pcm_out = self.istft.process(spec * tfmask)

            return pcm_out

        def reset(self):
            """Reset the internal state of the model."""
            self.feat_extractor.reset()
            self.specs = []
            self.istft.reset()
            if self.num_lookahead > 0:
                _, z_spec = feat_extractor(
                    np.zeros(hop_size, dtype=np.float32))  # Warm up the feature extractor
                for i in range(self.num_lookahead):
                    self.specs.append(z_spec.copy())
            if hasattr(self, 'dc_remover'):
                self.dc_remover.reset()

    # 4. Run the AudioShowClass
    se_model = SEModel(
        model_tflite=model_tflite,
        feat_extractor=feat_extractor,
        stats=stats,
        num_lookahead=params.train.num_lookahead
    )

    aud_handle = AudioShowClass(
        record_seconds=15,
        non_stop=True,
        proc_st=se_model,
        reset_st=se_model.reset,
        frame_size=hop_size,
        title="SoundKit SE Demo",
    )
     