from pathlib import Path
import tensorflow as tf
from ...utils.tflite_convert import tflite_convert, warp_tf_model   
from ...defines import SKTaskParams
from ...utils.download_tf_model import build_model, load_model_checkpoint
from ...utils.tf_copy_model import copy_model_weights

def export(params: SKTaskParams):
    """Export SE task model with given parameters.

    Args:
        params (HKTaskParams): Task parameters
    """
    params_export = params.export

    model_dir = f"{params.train['path']['models_trained']}/{params.name}"

    batchsize_train = params.train['batchsize']
    batchsize = 1
    dim_feat = params.train['feature']['bins']

    # 1.1. Build the model
    # Load from YAML file
    model_train = build_model(
        params, batchsize=batchsize_train, dim_feat=dim_feat)
    load_model_checkpoint(
        model_train, params_export['epoch_loaded'], model_dir)

    for v in model_train.trainable_variables:
        z = v.numpy().flatten()
        print(v.name, v.shape, z[1:5])
    

    model = build_model(
        params, batchsize=batchsize, dim_feat=dim_feat, export=True)
    copy_model_weights(model_dst=model, model_src=model_train)

    model_wrap = warp_tf_model(
        model,
        time_steps=1,
        dim_feat=dim_feat)
    tflite_fp16_model = tflite_convert(
        model_wrap,
        dtype='int16',
        path_tflite=f'{params.export["tflite_dir"]}/{params.name}.tflite',)