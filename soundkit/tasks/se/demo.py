import os
import logging
import subprocess
import shutil
from pathlib import Path
from .export import export
from ...defines import SKTaskParams
from ...utils.generate_feature_c_files import generate_feature_c_files
from ...utils.calculate_feat_stats import load_feat_stats
# === Configure Logging ===
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
    # === Setup Variables ===

    current_dir = Path.cwd().resolve()
    log.info(f"🔧 Current working directory: {current_dir}")

    tflite_filename_src = f"{params.name}.tflite"
    tflite_filename = "net.tflite"

    tflm_version = "ns_tflm_v1_0_0"

    evb_src_tflm_dir = Path(params.demo['evb_dir']) / 'src' # "tflm"

    # === Download neuralSPOT ===
    repo_url = "https://github.com/AmbiqAI/neuralSPOT.git"
    neuralSPOT = "neuralSPOT"
    neuralspot_path = Path(f"../{neuralSPOT}").resolve()
    if not os.path.exists(neuralspot_path):
        subprocess.run(["git", "clone", repo_url, neuralspot_path], check=True)
        log.info(f"📦 Cloned {neuralSPOT} to {neuralspot_path}")
    else:
        log.info(f"✅ {neuralSPOT} already exists at {neuralspot_path}")

    # === Generate Feature C Files ===
    log.info("🧪 Generating feature C files")

    model_dir = f"{params.train['path']['models_trained']}/{params.name}"
    

    # === Generate C Code STFT Window ===
    from ...utils.converter_fix_point import fakefix_tf, int2str_array
    from ...utils.tf_stft import gen_stft_win
    
    stft_win_name='stft_win_coeff'
    win_coeff = gen_stft_win(
        win_size=params.data['signal']['frame_size'],
        hop=params.data['signal']['hop_size'])
    win_coeff = fakefix_tf(win_coeff, 16, 15)
    c_code = int2str_array(stft_win_name, win_coeff.numpy()*32768, nbits=16)
    c_code = f"// stft window_coeff (framesize={params.data['signal']['frame_size']}, hopsize={params.data['signal']['hop_size']})\n" + c_code
    c_code = '#include <stdint.h>\n\n' + c_code
    Path(f"{evb_src_tflm_dir}/{stft_win_name}.c").write_text(c_code)


    # === Generate C Code Filter Banks ===
    import tensorflow as tf
    from ...utils.feature_utils import FeatureExtractor
    from ...utils.mel import gen_mel_c
    
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

    # === Generate feature statstics ===
    stats_name = 'stats.pkl'
    stats = load_feat_stats(
        dir=model_dir,
        stats_name=stats_name)

    generate_feature_c_files(
        file_name="def_nn3_se",
        param_struct_name="params_nn3_se",
        dir=evb_src_tflm_dir,
        feature_mean=stats['nMean_feat'],
        feature_std=stats['nInvStd'],
        sampling_rate=params.data['signal']['sampling_rate'],
        fftsize=params.data['signal']['fft_size'],
        winsize_stft=params.data['signal']['frame_size'],
        hopsize_stft=params.data['signal']['hop_size'],
        num_mfltrBank=params.train['feature']['bins'],
        is_dcrm=int(params.data['signal']['dc_removal']),
        pre_gain_q1=params.demo['pre_gain'],
        lookahead=params.train['num_lookahead'],
        stft_win_coeff_name=stft_win_name,
        filterbank_name=filterbank_name,
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

    subprocess.run(["python", "-m", "venv", ".venv"], check=True)
    subprocess.run([".venv/bin/pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([".venv/bin/pip", "install", "."], check=True)

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
