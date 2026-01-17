"""Export KWS task model."""
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from  soundkit.utils.tflite_convert import tflite_convert, warp_tf_model   
from soundkit.defines import SKTaskParams
from soundkit.utils.download_tf_model import build_model, load_model_checkpoint
from soundkit.utils.tf_copy_model import copy_model_weights
from soundkit.utils.feature_utils import FeatureExtractor
from .datasets import create_dataset
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
    # Prepare calibration data for quantization if needed
    if params.export.calibration_samples is not None:
        tfrecord_list = {
            'train': 
                Path(params.data['path_tfrecord']) / params.data['tfrecord_datalist_name']['train'],
            'val': 
                Path(params.data['path_tfrecord']) / params.data['tfrecord_datalist_name']['val'],
        }
        truncate_samples = int(
            params.train['truncate_time'] * params.data.signal.sampling_rate) \
                if params.train['truncate_time'] is not None else None

        ds_train, *_ = create_dataset(
            tfrecord_list['train'],
            batchsize=params.train['batchsize'],
            is_shuffle=True,
            num_per_epoch_files=params.train.num_per_epoch_files.train,
            truncate_samples=truncate_samples,
        )
        # 1. Flatten the dataset and extract the exact number of required calibration samples.
        # We discard additional labels/metadata using a map, keeping only the raw features.
        ds_collected = ds_train.unbatch() \
                            .take(params.export.calibration_samples) \
                            .map(lambda x, *args: x) \
                            .batch(params.export.calibration_samples)

        # 2. Materialize the dataset into a single Tensor.
        # next(iter()) is used here to efficiently pull the first (and only) batch 
        # into memory as a TensorFlow constant.
        audio_sn_tf = next(iter(ds_collected))

        # 3. Compute features using the GPU-accelerated extractor.
        # We process the entire calibration set as one batch and convert to a 
        # NumPy array only at the final step for downstream compatibility.
        data_calibration = feat_extractor(audio_sn_tf)[0].numpy()

        # for complex features-handling, split real and imaginary parts
        if np.iscomplexobj(data_calibration):
            data_calibration = np.stack(
                [np.real(data_calibration), np.imag(data_calibration)],
                axis=-1)

    else:
        data_calibration = None
    tflite_fp16_model = tflite_convert(
        model_wrap,
        dtype=params.export.dtype,
        path_tflite=path_tflite,
        data_calibration=data_calibration,)

    log.info(f"Exported model to {path_tflite}")
