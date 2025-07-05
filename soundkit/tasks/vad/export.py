from pathlib import Path
import tensorflow as tf
from ...utils.tflite_convert import tflite_convert, warp_tf_model
from ...defines import SKTaskParams
from ...utils.download_tf_model import build_model, load_model_checkpoint
from ...utils.tf_copy_model import copy_model_weights
from ...utils.feature_utils import FeatureExtractor

def export(params: SKTaskParams):
    """Export VAD task model with given parameters.

    Args:
        params (HKTaskParams): Task parameters
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
    model_train = build_model(
        params,
        batchsize=batchsize_train,
        dim_feat=dim_feat,
        time_steps = params.data['target_length_in_secs'] * params.data.signal.sampling_rate //  params.train.feature.hop_size)

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
    tflite_fp16_model = tflite_convert(
        model_wrap,
        dtype='int16',
        path_tflite=f'{params.export["tflite_dir"]}/{params.name}.tflite',)

    print(f"Exported model to {params.export['tflite_dir']}/{params.name}.tflite")

def export1(params: SKTaskParams):
    """Export VAD task model with given parameters.

    Args:
        params (HKTaskParams): Task parameters
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


    model_train = build_model(
        params,
        batchsize=batchsize_train,
        dim_feat=dim_feat,
        time_steps = params.data['target_length_in_secs'] * params.data.signal.sampling_rate //  params.train.feature.hop_size)
    
    load_model_checkpoint(
        model_train, params_export['epoch_loaded'], checkpoint_dir)

    model = build_model(
        params,
        batchsize=batchsize,
        dim_feat=dim_feat,
        time_steps=1,
        export=True,
        new_nn=False)

    copy_model_weights(model_dst=model, model_src=model_train)

    inputs_x = tf.keras.Input(shape=[1, dim_feat], batch_size=batchsize, dtype=tf.float32, name="x_input")
    inputs_reset = tf.keras.Input(shape=(), batch_size=batchsize, dtype=tf.float32, name="reset_input")
    output = model(inputs_x, reset_input=inputs_reset)
    model_export = tf.keras.Model(inputs=[inputs_x, inputs_reset], outputs=output)
    from .model import tflite_conversion
    tflite_conversion(
        model_export,
        timesteps=1,
        input_dim=dim_feat,
        path_tflite=f'{params.export["tflite_dir"]}/{params.name}.tflite',
        dtype="int16")


    print(f"Exported model to {params.export['tflite_dir']}/{params.name}.tflite")
