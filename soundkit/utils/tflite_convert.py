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
        qbit: int = 8,
        path_tflite: str = "./tflite/nnse.tflite",
        converter_with_reset: bool = True):
    """tflite converter"""
    os.makedirs('tflite', exist_ok=True)

    def dataset_example():
        shapes = model._feed_input_shapes
        shape_inputs = shapes[0]
        for i in range(100):
            x = np.random.uniform(
                -32768 / 2**qbit, 32767 / 2**qbit,
                size=shape_inputs).astype(np.float32)
            if converter_with_reset:
                reset_value = 0.0 if i < 80 else 1.0  # 80% no-reset, 20% reset
                reset = np.array([reset_value], dtype=np.float32)
                yield {"x_input": x, "reset_input": reset}
            else:
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

    
    file=open('soundkit/tasks/vad2/evb/src/input_output.cc', 'w')
    file.write('#include <stdint.h>\n')
    x = tf.random.uniform(model._feed_input_shapes[0], dtype=tf.float32)

    interpreter = tf.lite.Interpreter(
        model_content=net_tflite)
    interpreter.allocate_tensors()  # Needed before execution!

    # Get input and output tensors.
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    if dtype in ("int8", "int16"):
        input_scale, input_zero_point = input_details["quantization"]
        input_data = x.numpy() / input_scale + input_zero_point

        if dtype == "int8":
            input_data = input_data.astype(np.int8)
            file.write('int8_t inputs[]={\n')

        elif dtype == "int16":
            input_data = input_data.astype(np.int16)
            file.write('int16_t inputs[]={\n')

        input_data_flat = input_data.flatten()
        for d in input_data_flat:
            file.write(f'{d}, ')
        file.write('};\n')
    print(input_data)
    interpreter.set_tensor(
            input_details['index'],
            input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])

    if dtype == "int8":
        file.write('int8_t outputs[]={\n')
    elif dtype == "int16":
        file.write('int16_t outputs[]={\n')
    for d in output_data.flatten():
        file.write(f'{d}, ')
    file.write('};\n')


    return net_tflite

def warp_tf_model(
        model,
        dim_feat=257,
        time_steps=1,
        batch_size=1,
        converter_with_reset: bool = True):
    """ Convert the model to tflite format """
    # states = model.make_states(batchsize=1)

    input_shape=(time_steps, dim_feat)

    inputs_feat = tf.keras.Input(
        shape=input_shape,
        batch_size=batch_size,
        name='x_input') # batch_size fixed to 1
    if converter_with_reset:
        reset_input = tf.keras.Input(
            shape=(),
            batch_size=batch_size,
            dtype=tf.float32,
            name='reset_input')
        outputs= model(inputs_feat, reset_input=reset_input)
        model_wrap = tf.keras.Model(
        inputs=[
            inputs_feat,
            reset_input
            ],
        outputs=[
            outputs,
            ])
    else:
        outputs = model(inputs_feat)
        model_wrap = tf.keras.Model(
        inputs=[
            inputs_feat],
        outputs=[
            outputs,
            ])

    return model_wrap


def test_conv2d_group():
    from .unet1 import unet,encoder_unet, SeparableConv2D
    class MyNet(tf.keras.Model):
        def __init__(self, num_ch):
            super().__init__()

            self.conv = SeparableConv2D(
                filters=16,
                kernel_size=(2,3),
                strides=(1,2),
                num_channels_in=1,
                activation='relu',
                )
            # self.conv = tf.keras.layers.Conv2D(
            #     filters=16,
            #     kernel_size=(2,3),
            #     strides=(1,2),
            #     padding='valid',
            #     # groups=num_ch,
            #     activation='relu',
            #     use_bias=True,
            # )
            self.dense = tf.keras.layers.Dense(
                units=960,
                activation='relu',
                kernel_initializer='he_normal',
                bias_initializer='zeros',
            )
        def build(self, input_shape):
            pass

        def call(self, inputs):
            out = inputs
            # out = self.conv(out)
            # out = out[:,:,:,0]
            
            out = self.conv(inputs)

            return out

    dim_feat=257
    time_steps=1
    num_ch=1
    dtype='int16'

    nn_train = unet(
                dim_feat=dim_feat,
                kernel_size_time=2,
                batch_size=1,
                separable=True,
                unroll_rnn=True,
                num_chs= [1, 12, 12, 12, 12]
                # num_chs=[1, 2, 2, 2, 2],
                )
    # nn_train = MyNet(num_ch)
    file=open('tflite/input_output.cc', 'w')
    file.write('#include <stdint.h>\n')
    if num_ch != 0:
        x = tf.random.uniform((1, time_steps, dim_feat,num_ch), dtype=tf.float32)
    else:
        x = tf.random.uniform((1, time_steps, dim_feat), dtype=tf.float32)
    # x = x * 100
    # x = tf.reshape(a, (1, time_steps, dim_feat,num_ch))

    y = nn_train(x)
    print(y)
    nn = warp_tf_model(
        nn_train,
        time_steps=time_steps,
        dim_feat=dim_feat)
    # nn.trainable_variables[0].assign(tf.ones_like(nn.trainable_variables[0]))

    tflite_fp16_model = tflite_convert(
        nn,
        dtype=dtype,
        path_tflite='./tflite/test.tflite')
    interpreter = tf.lite.Interpreter(
        model_content=tflite_fp16_model)
    interpreter.allocate_tensors()  # Needed before execution!

    # Get input and output tensors.
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    if dtype in ("int8", "int16"):
        input_scale, input_zero_point = input_details["quantization"]
        input_data = x.numpy() / input_scale + input_zero_point

        if dtype == "int8":
            input_data = input_data.astype(np.int8)
            file.write('int8_t inputs[]={\n')

        elif dtype == "int16":
            input_data = input_data.astype(np.int16)
            file.write('int16_t inputs[]={\n')

        input_data_flat = input_data.flatten()
        for d in input_data_flat:
            file.write(f'{d}, ')
        file.write('};\n')
    print(input_data)
    interpreter.set_tensor(
            input_details['index'],
            input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])

    if dtype == "int8":
        file.write('int8_t outputs[]={\n')
    elif dtype == "int16":
        file.write('int16_t outputs[]={\n')
    for d in output_data.flatten():
        file.write(f'{d}, ')
    file.write('};\n')
    if dtype in ("int8", "int16"):
        output_scale, output_zero_point = output_details["quantization"]

        out = (output_data - output_zero_point).astype(np.float32) * output_scale

    print(output_data.flatten())
    print(out.shape)
    file.close()

def test_pre_train():


    dim_feat=257
    time_steps=1
    num_ch=0
    dtype='int16'

    file=open('tflite/input_output.cc', 'w')
    file.write('#include <stdint.h>\n')
    if num_ch != 0:
        x = tf.random.uniform((1, time_steps, dim_feat,num_ch), dtype=tf.float32)
    else:
        x = tf.random.uniform((1, time_steps, dim_feat), dtype=tf.float32)
    x = x * 100
    
    interpreter = tf.lite.Interpreter(
        model_path=f'./tflite/nnse_{dtype}.tflite')
    interpreter.allocate_tensors()  # Needed before execution!

    # Get input and output tensors.
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    if dtype in ("int8", "int16"):
        input_scale, input_zero_point = input_details["quantization"]
        input_data = x.numpy() / input_scale + input_zero_point

        if dtype == "int8":
            input_data = input_data.astype(np.int8)
            file.write('int8_t inputs[]={\n')

        elif dtype == "int16":
            input_data = input_data.astype(np.int16)
            file.write('int16_t inputs[]={\n')

        input_data_flat = input_data.flatten()
        for d in input_data_flat:
            file.write(f'{d}, ')
        file.write('};\n')
    print(input_data)
    interpreter.set_tensor(
            input_details['index'],
            input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])

    if dtype == "int8":
        file.write('int8_t outputs[]={\n')
    elif dtype == "int16":
        file.write('int16_t outputs[]={\n')
    for d in output_data.flatten():
        file.write(f'{d}, ')
    file.write('};\n')
    if dtype in ("int8", "int16"):
        output_scale, output_zero_point = output_details["quantization"]

        out = (output_data - output_zero_point).astype(np.float32) * output_scale

    print(output_data.flatten())
    print(out.shape)
    file.close()

if __name__ == "__main__":
    test_conv2d_group()
    # test_pre_train()
