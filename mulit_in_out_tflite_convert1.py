import tensorflow as tf
import numpy as np
import os
from pathlib import Path
# -------------------------------
# Define the Conv+LSTM model
# -------------------------------


class ConvLSTMHybridModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(
            filters=100,
            kernel_size=(6, 40),
            activation='tanh',
            padding='valid'
        )
        self.lstm = tf.keras.layers.LSTM(
            units=100,
            return_sequences=True,
            return_state=True,
            unroll=True,
            activation='tanh',
            recurrent_activation='sigmoid'
        )
        self.dense = tf.keras.layers.Dense(64, activation='linear')

    def call(self, input_0, cnn_state, h_state, c_state):
        x = tf.concat([input_0, cnn_state], axis=1)  # [1, 185, 40]

        cnn_state=tf.identity(x[:,-5:,:])  # Ensure cnn_state is not None
        x = tf.expand_dims(x, axis=-1)  # [1, 185, 40, 1]
        x = self.conv(x)  # [1, 180, 1, 32]
        x = tf.squeeze(x, axis=2)  # [1, 180, 32]
        lstm_out, h, c = self.lstm(x, initial_state=[h_state, c_state])
        output = self.dense(lstm_out)  # [1, 180, 64]
        return output, cnn_state, h, c



def tflite_convert(
        model: tf.keras.Model,
        input_shapes: list = [[1, 40], [5, 40], [100], [100]],
        input_names: list = ["input_0", "cnn_state", "h_state", "c_state"],
        input_ranges: list = [[-1, 1], [-1, 1], [-1, 1], [-1, 1]],
        output_names: list = ["output", "cnn_state", "h_state", "c_state"],
        dtype: str = "int8",
        path_tflite: str = "./tflite/nnse.tflite"):
    """tflite converter"""


    inputs = []
    for i in range(len(input_shapes)):
        inputs.append(tf.keras.Input(shape=input_shapes[i], name=input_names[i]))

    outputs = model(*inputs)
    keras_model = tf.keras.Model(
        inputs=inputs,
        outputs=list(outputs)  # output, cnn_state_new, h, c
    )

    # -------------------------------
    # Define the serving signature
    # -------------------------------
    input_signature = []
    for shape, name in zip(input_shapes, input_names):

        shape= [1] + shape  # Add batch dimension
        input_signature.append(tf.TensorSpec(shape=shape, dtype=tf.float32, name=name))


    @tf.function(
        input_signature=input_signature,
    )
    def serving_fn(*inputs):
        outputs = keras_model(inputs)
        outputs_dict = dict(zip(output_names, outputs))

        return outputs_dict

    # Save model
    tf.saved_model.save(model, "quant_model", signatures={"serving_default": serving_fn})

    # -------------------------------
    # Calibration data for quantization
    # -------------------------------
    def representative_data_gen():
        for _ in range(100):
            
            yield {
                name: tf.random.uniform(
                    [1] + shape,
                    minval=rng[0],
                    maxval=rng[1],
                    dtype=tf.float32)
                for name, shape, rng in zip(input_names, input_shapes, input_ranges)
            }

    # -------------------------------
    # TFLite Conversion
    # -------------------------------
    converter = tf.lite.TFLiteConverter.from_saved_model("quant_model")
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.inference_input_type = tf.int16
    converter.inference_output_type = tf.int16
   
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(path_tflite), exist_ok=True)
    
    path_tflite_b=Path(path_tflite)
    path_tflite_b.write_bytes(tflite_model)
    os.system(f"xxd -i {path_tflite_b} > {os.path.dirname(path_tflite)}/model_data_{dtype}.c")
    return tflite_model


if __name__ == "__main__":
    timesteps = 1

    model = ConvLSTMHybridModel()
    input_ranges=[
        [-32768.0 / 2**8, 32767.0 / 2**8],
        [-32768.0 / 2**8, 32767.0 / 2**8],
        [-1, 1],
        [-32768.0 / 2**8, 32767.0 / 2**8]]
    path_tflite="./tflite/quant_model_int16x8.tflite"
    tflite_convert(model=model,
                input_shapes=[[timesteps, 40], [5, 40], [100], [100]],
                input_names=["input_0", "cnn_state", "h_state", "c_state"],
                input_ranges=input_ranges,
                output_names=["output", "cnn_state", "h_state", "c_state"],
                dtype="int16",
                path_tflite=path_tflite)
    # -------------------------------
    # Confirm I/O Shapes
    # -------------------------------
    interpreter = tf.lite.Interpreter(model_path=path_tflite)
    interpreter.allocate_tensors()

    print("\n📥 Input Tensors:")

    for inp in interpreter.get_input_details():
        print(f"  - {inp['name']}, shape={inp['shape']}, dtype={inp['dtype']}")

    print("\n📤 Output Tensors:")
    for out in interpreter.get_output_details():
        print(f"  - {out['name']}, shape={out['shape']}, dtype={out['dtype']}")
    import pdb; pdb.set_trace()