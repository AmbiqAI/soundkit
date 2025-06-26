import tensorflow as tf
import numpy as np
import os
from pathlib import Path

# ------------------------------------------------
# 1. ConvLSTM model with resource states + reset
# ------------------------------------------------
class ConvLSTMHybridModel_ID(tf.keras.Model):
    """ConvLSTM model with internal states and reset functionality."""
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
        self.fc1 = tf.keras.layers.Dense(100, activation='tanh')

        self.fc2 = tf.keras.layers.Dense(64, activation='linear')

        self.cnn_state = tf.Variable(
            initial_value=tf.zeros([1, 5, 40], dtype=tf.float32),
            trainable=False,
            name="cnn_state"
        )
        self.h_state = tf.Variable(
            initial_value=tf.zeros([1, 100], dtype=tf.float32),
            trainable=False,
            name="h_state"
        )
        self.c_state = tf.Variable(
            initial_value=tf.zeros([1, 100], dtype=tf.float32),
            trainable=False,
            name="c_state"
        )

    def call(self, input0, reset=0):
        """Forward pass with reset functionality."""
        x = tf.concat([self.cnn_state, input0], axis=1)  # [1, 185, 40]

        cnn_state=tf.identity(x[:,-5:,:])  # Ensure cnn_state is not None
        self.cnn_state.assign(cnn_state - reset*cnn_state)  # Update cnn_state

        x = tf.expand_dims(x, axis=-1)  # [1, 185, 40, 1]
        x = self.conv(x)  # [1, 180, 1, dim_feat]
        x = tf.squeeze(x, axis=2)  # [1, 180, dim_feat]
        lstm_out, h, c = self.lstm(x, initial_state=[self.h_state, self.c_state])

        self.h_state.assign(h - reset*h)  # Update h_state
        self.c_state.assign(c - reset*c)  # Update c_state

        output = self.fc1(lstm_out)  # [1, 180, 100]
        output = self.fc2(output)  # [1, 180, 64]
        return output

class ConvLSTMHybridModel_VAD(tf.keras.Model):
    """ConvLSTM model with internal states and reset functionality for VAD."""
    def __init__(self):
        super().__init__()

        self.fc_in = tf.keras.layers.Dense(
            28,
            activation='tanh')

        self.conv = tf.keras.layers.Conv2D(
            filters=28,
            kernel_size=(6, 28),
            activation='tanh',
            padding='valid'
        )

        self.lstm = tf.keras.layers.LSTM(
            units=28,
            return_sequences=True,
            return_state=True,
            unroll=True,
            activation='tanh',
            recurrent_activation='sigmoid'
        )

        self.fc_1 = tf.keras.layers.Dense(28, activation='relu')
        self.fc_2 = tf.keras.layers.Dense(28, activation='relu')
        self.fc_3 = tf.keras.layers.Dense(2, activation='linear')

        self.cnn_state = tf.Variable(
            initial_value=tf.zeros([1, 5, 28], dtype=tf.float32),
            trainable=False,
            name="cnn_state"
        )
        self.h_state = tf.Variable(
            initial_value=tf.zeros([1, 28], dtype=tf.float32),
            trainable=False,
            name="h_state"
        )
        self.c_state = tf.Variable(
            initial_value=tf.zeros([1, 28], dtype=tf.float32),
            trainable=False,
            name="c_state"
        )

    def call(self, input0, reset=1.0):
        """Forward pass with reset functionality."""
        x = self.fc_in(input0)  # [1, 180, 28]
        x = tf.concat([self.cnn_state, x], axis=1)  # [1, 185, dim_feat]
        cnn_state=tf.identity(x[:,-5:,:])
        self.cnn_state.assign(cnn_state - reset*cnn_state)  # Update cnn_state
        
        x = tf.expand_dims(x, axis=-1)  # [1, 180, 28, 1]
        x = self.conv(x)  # [1, 180, 1, 28]
        x = tf.squeeze(x, axis=2)  # [1, 180, 28]
        lstm_out, h, c = self.lstm(x, initial_state=[self.h_state, self.c_state])

        self.h_state.assign(h - reset*h)  # Update h_state
        self.c_state.assign(c - reset*c)  # Update c_state

        output = self.fc_1(lstm_out)
        output = self.fc_2(output)
        output = self.fc_3(output)  # [1, 180, 2
        return output

# ------------------------------------------------
# 2. TFLite converter (16x8 with correct reset)
# ------------------------------------------------
def tflite_convert_16x8(
        model,
        input_shapes,
        input_range,
        path_tflite):
    """Convert the ConvLSTM model to TFLite with 16x8 quantization."""
    @tf.function(
        input_signature=[
            tf.TensorSpec(input_shapes, tf.float32, name="input_0"),
            tf.TensorSpec([1], tf.float32, name="reset")
        ]
    )
    def serving_fn(input_0, reset):
        return {"output": model(input_0, reset)}

    tf.saved_model.save(model, "quant_model", signatures={"serving_default": serving_fn})

    def representative_data_gen():
        for i in range(100):
            yield {
                "input_0": tf.random.uniform(
                    input_shapes,
                    minval=input_range[0],
                    maxval=input_range[1]),
                "reset": tf.constant([1.0 if i % 10 == 0 else 0.0], dtype=tf.float32)
            }

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
    Path(path_tflite).write_bytes(tflite_model)
    os.system(f"xxd -i {path_tflite} > {os.path.dirname(path_tflite)}/model_data_int16x8.c")

    return tflite_model

def run_example(input_shape, tflite_model_path):

    print("\n▶️ Running inference example...")

    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Inspect input names and indexes
    for i, inp in enumerate(input_details):
        print(f"Input {i}: name={inp['name']}, index={inp['index']}, shape={inp['shape']}, dtype={inp['dtype']}")

    # Create input_0 (quantized if needed)
    input0_fp = np.random.uniform(-1, 1, size=input_shape).astype(np.float32)
    reset_fp = np.array([-1.0], dtype=np.float32)  # or [0.0] for no reset

    # Quantize if needed

    if input_details[1]['dtype'] == np.int16:
        scale, zero_point = input_details[1]['quantization']
        input0 = np.round(input0_fp / scale).astype(np.int16)
    else:
        input0 = input0_fp

    if input_details[0]['dtype'] == np.int16:
        scale, zero_point = input_details[0]['quantization']

        reset = np.round(reset_fp / scale).astype(np.int16)
        print("🔢 Reset quantized:", reset)
    else:
        reset = reset_fp

    # Set inputs
    interpreter.set_tensor(input_details[0]['index'], reset)
    interpreter.set_tensor(input_details[1]['index'], input0)

    interpreter.invoke()

    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])

    print("\n✅ Inference completed. Output shape:", output.shape)
    print("🔢 Output (first element):\n", output[0, 0])  # just sample first timestep

# Call it


# ------------------------------------------------
# 3. Run and Inspect
# ------------------------------------------------
if __name__ == "__main__":

    task="vad" # "id" for ConvLSTMHybridModel_ID, "vad" for ConvLSTMHybridModel_VAD

    if task == "id":
        model = ConvLSTMHybridModel_ID()
        input_shapes = [1, 1, 40]  # input_0 shape

    else:
        model = ConvLSTMHybridModel_VAD()
        input_shapes = [1, 1, 257]  # input_0 shape
    
    path_tflite = "./tflite/quant_model_int16x8.tflite"

    print("🔄 Converting model to TFLite...")
    tflite_model = tflite_convert_16x8(
        model,
        input_shapes=input_shapes,
        input_range=[-1.0, 1.0],
        path_tflite=path_tflite
    )

    print("📦 Saved to:", path_tflite)

    interpreter = tf.lite.Interpreter(model_path=path_tflite)
    interpreter.allocate_tensors()

    print("\n📥 Input Tensors:")
    for i in interpreter.get_input_details():
        print(f"  - {i['name']}, shape={i['shape']}, dtype={i['dtype']}")

    print("\n📤 Output Tensors:")
    for o in interpreter.get_output_details():
        print(f"  - {o['name']}, shape={o['shape']}, dtype={o['dtype']}")

    run_example(input_shapes, "./tflite/quant_model_int16x8.tflite")
