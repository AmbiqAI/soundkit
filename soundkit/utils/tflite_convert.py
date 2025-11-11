""" Convert the model to tflite format """
import os
from pathlib import Path
from typing import Callable
import tensorflow as tf
import numpy as np

def convert_model(
        model: tf.keras.Model,
        dataset_gen: Callable,
        dtype: str = "int8"):
    """ tflite converter

    Args:
        model (tf.keras.Model): _description_
        dataset_gen (Callable): _description_
        nbit (int, optional): _description_. Defaults to 8.

    Returns:
        bytes: _description_
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if dtype=="int8":
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS
            ]  # enable TensorFlow ops.
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.representative_dataset = dataset_gen
    elif dtype=="int16":
        # converter.target_spec.supported_types = [tf.int8]
        # converter._experimental_full_integer_quantization_bias_type = tf.int32
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS, # enable TensorFlow ops.
            ]
        converter.inference_input_type  = tf.int16
        converter.inference_output_type = tf.int16
        converter.representative_dataset = dataset_gen
    elif dtype=='float16':
        converter.target_spec.supported_types = [tf.float16]
    else:
        converter.target_spec.supported_types = [
            tf.float32]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
            ]
        converter.inference_input_type = tf.float32

    return converter.convert()

def tflite_convert(
        model: tf.keras.Model,
        dtype: str = "int8",
        path_tflite: str = "./tflite/nnse.tflite",
        nbits: int = 16,
        qbits: int = 8):
    """tflite converter"""

    os.makedirs('tflite', exist_ok=True)

    min_val = -2**(nbits-1) / 2**qbits
    max_val = (2**(nbits-1) - 1) / 2**qbits

    def dataset_example():
        shapes = model._feed_input_shapes
        shape_inputs = shapes[0]
        for _ in range(100):
            x = np.random.uniform(
                min_val, max_val,
                size=shape_inputs).astype(np.float32)
            yield {"x_input": x}

    model.summary()

    net_tflite = convert_model(
        model,
        dataset_example,
        dtype=dtype)

    os.makedirs(os.path.dirname(path_tflite), exist_ok=True)

    path_tflite_b=Path(path_tflite)
    path_tflite_b.write_bytes(net_tflite)
    os.system(f"xxd -i {path_tflite_b} > {os.path.dirname(path_tflite)}/model_data_{dtype}.c")
    return net_tflite

def warp_tf_model(
        model,
        dim_feat=257,
        time_steps=1,
        batch_size=1,
        is_complex=False):
    """ Convert the model to tflite format """
    # states = model.make_states(batchsize=1)
    if is_complex:
        input_shape = (time_steps, dim_feat, 2)
    else:   
        input_shape=(time_steps, dim_feat)

    inputs_feat = tf.keras.Input(
        shape=input_shape,
        batch_size=batch_size,
        name='x_input') # batch_size fixed to 1

    outputs= model(inputs_feat)

    model_wrap = tf.keras.Model(
        inputs=[
            inputs_feat,
            ],
        outputs=[
            outputs,
            ])

    return model_wrap
