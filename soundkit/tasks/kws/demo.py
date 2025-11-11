""" KWS Demo for EVB and PC"""
import os
import logging
import subprocess
import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
from soundkit.utils.tflite_convert import tflite_convert, warp_tf_model
from soundkit.defines import SKTaskParams
from soundkit.utils.download_tf_model import build_model, load_model_checkpoint
from soundkit.utils.tf_copy_model import copy_model_weights
from soundkit.utils.feature_utils import FeatureExtractor
from soundkit.utils.np_feature_utils import FeatureExtractor_np
from soundkit.utils.pyaudio_animation import AudioShowClass
from soundkit.utils.calculate_feat_stats import load_feat_stats
from soundkit.utils.TFLiteAudioModel import TFLiteAudioModel
from soundkit.utils.generate_feature_c_files import generate_feature_c_files
from soundkit.utils.basic_dsp import DCRemover
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
    if params.demo.platform == 'evb':
        demo_evb(params)
    elif params.demo.platform == 'pc':
        demo_pc(params)
    else:
        raise ValueError(f"Unsupported platform: {params.demo.platform}. Supported platforms are 'evb' and 'pc'.")

def demo_evb(params: SKTaskParams):
    """
    Deploy a TFLite model to neuralSPOT and install dependencies.

    Args:
        params (SKTaskParams): Task parameters
    """
    # === Setup Variables ===

    current_dir = Path.cwd().resolve()
    log.info(f"🔧 Current working directory: {current_dir}")

    tflite_filename_src = f"{params.name}_{params.export['dtype']}.tflite"
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

    # === Generate Feature C Files ===
    log.info("🧪 Generating feature C files")

    checkpoint_dir = f"{params.train['path']['checkpoint_dir']}"

    # === Generate C Code STFT Window ===
    from ...utils.converter_fix_point import fakefix_tf, int2str_array
    from ...utils.tf_stft import gen_stft_win
    feat_params = params.train['feature']
    stft_win_name='stft_win_coeff'
    win_coeff = gen_stft_win(
        win_size=feat_params['frame_size'],
        hop=feat_params['hop_size'])
    win_coeff = fakefix_tf(win_coeff, 16, 15)
    c_code = int2str_array(stft_win_name, win_coeff.numpy()*32768, nbits=16)
    c_code = f"// stft window_coeff (framesize={feat_params['frame_size']}, hopsize={feat_params['hop_size']})\n" + c_code
    c_code = '#include <stdint.h>\n\n' + c_code
    Path(f"{evb_src_tflm_dir}/{stft_win_name}.c").write_text(c_code)


    # === Generate C Code Filter Banks ===
    import tensorflow as tf
    from ...utils.mel import gen_mel_c

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
        task=params.project
    )

    # === Define Key Paths ===
    src_tflite_path = Path(params.demo['tflite_dir']) / tflite_filename_src

    tools_dir = Path(f"../{neuralSPOT}/tools").resolve()
    dst_tflite_path = tools_dir / tflite_filename
    neuralspot_root = Path(f"../{neuralSPOT}").resolve()

    # === export TFLite File ===
    log.info(f"🧪 Exporting TFLite model from {src_tflite_path}")
    params.export['epoch_loaded'] = params.demo['epoch_loaded']
    params.export['tflite_dir'] = params.demo['tflite_dir']
    export(params)

    # === Copy TFLite File to neuralSPOT/tools ===

    log.info(f"📦 Copying TFLite to {dst_tflite_path}")
    tools_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_tflite_path, dst_tflite_path)

    # === Setup venv and install ===
    log.info(f"🔧 Setting up virtual environment at {neuralspot_root}")
    os.chdir(neuralspot_root)
    (neuralspot_root / "projects/autodeploy").mkdir(parents=True, exist_ok=True)

    subprocess.run(["uv", "python", "pin", "3.12.11"], check=True)
    subprocess.run(["uv", "sync"], check=True)
    # subprocess.run(["python", "-m", "venv", ".venv"], check=True)
    # subprocess.run([".venv/bin/pip", "install", "--upgrade", "pip"], check=True)
    # subprocess.run([".venv/bin/pip", "install", "."], check=True)

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

def demo_pc(params: SKTaskParams):
    """Export VAD task model with given parameters.

    Args:
        params (SKTaskParams): Task parameters
    """
    params_export = params.export

    checkpoint_dir = f"{params.train['path']['checkpoint_dir']}"

    batchsize_train = params.train['batchsize']
    batchsize = 1

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
    dim_feat = feat_extractor.dim_feat

    # 1.1. Build the model
    # Load from YAML file
    time_steps = int(params.data['target_length_in_secs'] * params.data.signal.sampling_rate) //  params.train.feature.hop_size
    model_train = build_model(
        params,
        batchsize=batchsize_train,
        dim_feat=dim_feat,
        time_steps = time_steps,)

    load_model_checkpoint(
        model_train, params_export['epoch_loaded'], checkpoint_dir)

    model = build_model(
        params,
        batchsize=batchsize,
        dim_feat=dim_feat,
        time_steps=1,
        export=True)
    copy_model_weights(model_dst=model, model_src=model_train)

    model_wrap = warp_tf_model(
        model,
        time_steps=1,
        dim_feat=dim_feat)

    dtype='float32'

    tflite_fp16_model = tflite_convert(
        model_wrap,
        dtype=dtype,
        path_tflite=f'{params.export["tflite_dir"]}/{params.name}.tflite',)

    interpreter = tf.lite.Interpreter(
        model_content=tflite_fp16_model)
    interpreter.allocate_tensors()  # Needed before execution!

    if params.train.standardization:
        stats = load_feat_stats(checkpoint_dir, 'stats.pkl')
    else:
        stats = None

    model_tflite = TFLiteAudioModel(
        interpreter=interpreter,
        dtype=dtype,
    )

    class KwsModel:
        """KWS model class in tflite for processing audio input and returning VAD output."""
        def __init__(
                self,
                model_tflite: TFLiteAudioModel,
                feat_extractor: FeatureExtractor_np,
                stats: dict | None = None):
            self.model_tflite = model_tflite
            self.stats = stats
            self.feat_extractor = feat_extractor
            if params.data.signal.dc_removal:
                self.dc_remover = DCRemover()
            self.reset()

        def reset(self):
            """Reset the model state if needed."""
            # This is a no-op for TFLite models, but can be overridden if needed.
            self.feat_extractor.reset()
            self.model_tflite.reset()
            if params.data.signal.dc_removal:
                self.dc_remover.reset()

        def __call__(self,
                     inputs: np.ndarray # input from microphone
                    ) -> np.ndarray: # output to AudioShowClass
            """Process input audio signal and return VAD output."""
            shape=inputs.shape
            inputs=inputs.flatten()
            if params.data.signal.dc_removal:
                inputs = self.dc_remover.process(inputs)
            features,_ = self.feat_extractor(inputs)

            if self.stats is not None:
                features = (features - self.stats['nMean_feat']) * self.stats['nInvStd']

            # input to the tflite model
            features = features.reshape((1, 1, -1)) # reshape to (batch_size, time_steps, dim_feat)
            outputs = self.model_tflite(features)
            outputs = outputs.flatten()
            if outputs[0] < outputs[1]:
                outputs = np.ones(160, dtype=np.float32)*0.95
            else:
                outputs = np.zeros(160, dtype=np.float32)
            outputs = outputs.reshape(shape)
            return outputs

    kws_model = KwsModel(
        model_tflite=model_tflite,
        feat_extractor=feat_extractor,
        stats=stats,
    )
    aud_handle = AudioShowClass(
        record_seconds=15,
        non_stop=True,
        proc_st=kws_model,
        reset_st=kws_model.reset,
        title="KWS",
    )
