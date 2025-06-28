"""Export ID task model with given parameters.
Args:
    params (HKTaskParams): Task parameters
"""
from soundkit.defines import SKTaskParams
from soundkit.utils.tflite_convert import tflite_convert, warp_tf_model
from soundkit.utils.download_tf_model import build_model, load_model_checkpoint
from soundkit.utils.tf_copy_model import copy_model_weights
from soundkit.utils.feature_utils import FeatureExtractor

def export(params: SKTaskParams):
    """Export ID task model with given parameters.

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
        time_steps = int(params.data['target_length_in_secs'] * 100))

    load_model_checkpoint(
        model_train, params_export['epoch_loaded'], checkpoint_dir)

    for v in model_train.trainable_variables:
        z = v.numpy().flatten()
        print(v.name, v.shape, z[1:5])


    # from .utils.tf_models_resetable import ConvLSTMHybridModel_ID

    # model = ConvLSTMHybridModel_ID()
    # import pdb; pdb.set_trace()
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

def export0(params: SKTaskParams):
    """Export ID task model with given parameters.

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
        time_steps = int(params.data['target_length_in_secs'] * 100))

    load_model_checkpoint(
        model_train, params_export['epoch_loaded'], checkpoint_dir)

    

    from .utils.tf_models_resetable import ConvLSTMHybridModel_ID, ConvLSTMHybridModel_VAD, tflite_convert_16x8

    model = ConvLSTMHybridModel_ID()
    import tensorflow as tf

    shape = [1, 1, 40]  # input_0 shape

    out = model(tf.random.uniform(shape, dtype=tf.float32))  # warmup
    # import pdb; pdb.set_trace()
    for u, v in zip(model_train.trainable_variables, model.trainable_variables):
        print(u.name, u.shape, v.name, v.shape)
        v.assign(u)

    tflite_convert_16x8(
        model=model,
        input_shapes=shape,
        input_range=[-32768.0 / 2**8, 32767.0 / 2**8],
        path_tflite=f'{params.export["tflite_dir"]}/{params.name}.tflite')

    # if 1:
    #     model_wrap = warp_tf_model(
    #         model,
    #         time_steps=1,
    #         dim_feat=dim_feat)
    #     tflite_fp16_model = tflite_convert(
    #         model_wrap,
    #         dtype='int16',
    #         path_tflite=f'{params.export["tflite_dir"]}/{params.name}.tflite',)
    # else:
    #     model_wrap = warp_tf_model_with_reset(
    #         model,
    #         time_steps=1,
    #         dim_feat=dim_feat)
    #     tflite_fp16_model = tflite_convert_with_reset(
    #         model_wrap,
    #         dtype='int16',
    #         path_tflite=f'{params.export["tflite_dir"]}/{params.name}.tflite',)
    # print(f"Exported model to {params.export['tflite_dir']}/{params.name}.tflite")
    