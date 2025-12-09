"""Export KWS task model."""
import logging
import tensorflow as tf
from  soundkit.utils.tflite_convert import tflite_convert, warp_tf_model   
from soundkit.defines import SKTaskParams
from soundkit.utils.download_tf_model import build_model, load_model_checkpoint
from soundkit.utils.tf_copy_model import copy_model_weights
from soundkit.utils.feature_utils import FeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)

def export(params: SKTaskParams):
    """Export SE task model with given parameters.

    Args:
        params (SKTaskParams): Task parameters
    """
    params_export = params.export

    checkpoint_dir = f"{params.train['path']['checkpoint_dir']}"

    batchsize_train = params.train['batchsize']
    batchsize = 1
    feat_extractor = FeatureExtractor(
        params=params,
        )
    dim_feat = feat_extractor.dim_feat

    # 1.1. Build the model
    # Load from YAML file

    time_steps = int(params.data['target_length_in_secs'] * params.data.signal.sampling_rate) //  params.train.feature.hop_size

    model_train = build_model(
        params,
        batchsize=batchsize_train,
        dim_feat=dim_feat,
        time_steps = time_steps)

    load_model_checkpoint(
        model_train,
        params_export['epoch_loaded'],
        checkpoint_dir,
        criterion_epoch="val_acc")

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

    path_tflite = f'{params.export["tflite_dir"]}/{params.name}_{params.export.dtype}.tflite'

    tflite_fp16_model = tflite_convert(
        model_wrap,
        dtype=params.export.dtype,
        path_tflite=path_tflite)

    log.info(f"Exported model to {path_tflite}")