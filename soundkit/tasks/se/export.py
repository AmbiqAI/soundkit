from pathlib import Path
import tensorflow as tf
from ...utils.tflite_convert import tflite_convert, warp_tf_model      
from ...defines import SKTaskParams
from ...utils.download_tf_model import build_model, load_model_checkpoint

def export(params: SKTaskParams):
    """Export SE task model with given parameters.

    Args:
        params (HKTaskParams): Task parameters
    """
    params_evaluate = params.evaluate
    num_lookahead = params.train['num_lookahead']
    signal = params.data['signal']
    model_dir = f'{params.job_dir}/models_trained/{params.name}'

    batchsize_train = params.train['batchsize']
    batchsize = 1
    dim_feat = params.train['feature']['bins']


    # 1.1. Build the model

    # Load from YAML file
    model_train = build_model(
        params, batchsize=batchsize_train, dim_feat=dim_feat)
    load_model_checkpoint(
        model_train, params_evaluate['epoch_loaded'], model_dir)

    def copy_model_weights(
            model_dst: tf.keras.Model,
            model_src: tf.keras.Model
        ) -> None:
        
        for s, d in zip(model_src.trainable_variables, model_dst.trainable_variables):
            # if s.name != d.name:
            #     raise ValueError(f"Model weights do not match: {s.name} != {d.name}")
            d.assign(s)

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
        path_tflite=f'{params.export["path_tflite"]}/{params.name}',)