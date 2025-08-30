import tensorflow as tf
import numpy as np

# --------- MODEL DEFINITION ---------
class StatefulLSTMModel_VAD(tf.keras.Model):
    def __init__(self, input_dim):
        super(StatefulLSTMModel_VAD, self).__init__()

        self.fc_in = tf.keras.layers.Dense(
            28,
            activation='tanh')
        
        
        self.conv2d = tf.keras.layers.Conv2D(
            filters=28,
            kernel_size=(6, 28),
            activation='tanh',
            padding='valid',
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


        self.input_state=None
        self.h_state = None
        self.c_state = None

    def build(self, input_shapes):
        """Build the model with dynamic batch size."""
        input_shape = input_shapes[0]  # x_input shape
        batch_size = input_shape[0]

        self.input_state = self.add_weight(
            shape=(batch_size, 5, 28),
            trainable=False,
            name="input_state",
            initializer="zeros"
        )
        self.h_state = self.add_weight(
            shape=(batch_size, 28),
            trainable=False,
            name="h_state",
            initializer="zeros"
        )
        self.c_state = self.add_weight(
            shape=(batch_size, 28),
            trainable=False,
            name="c_state",
            initializer="zeros"
        )

    def call(self, inputs, training=False):
        x_input, reset_input = inputs  # reset_input: [B, 1] int16
    
        x_input = self.fc_in(x_input)
        x_input = tf.concat([self.input_state, x_input], axis=1)  # Concatenate input with state
        input_state = tf.identity(x_input[:, -5:, :])  # Keep the last five time steps for input state
        self.input_state.assign(input_state * (1 - reset_input))   # Update input state

        x_input = tf.expand_dims(x_input, axis=-1)  # Add channel dimension for Conv2D
        
        x_input = self.conv2d(x_input)  # Apply Conv2D layer
        x_input = tf.squeeze(x_input, axis=-2)
        # Proceed with normal LSTM forward pass
        output, h_new, c_new = self.lstm(x_input, initial_state=[self.h_state, self.c_state])

        self.h_state.assign(h_new * (1 - reset_input))
        self.c_state.assign(c_new * (1 - reset_input))

        output = self.fc_1(output)  # Apply Dense layer
        output = self.fc_2(output)  # Apply Dense layer
        output = self.fc_3(output)  # Apply Dense layer

        return output

def tflite_conversion(
        export_model: tf.keras.Model,
        timesteps: int=1,
        input_dim: int=257,
        path_tflite: str="stateful_lstm_16x8_int_reset.tflite",
        dtype: str="int16"):
    """Convert the model to TFLite format."""


    # x_input = tf.keras.Input(shape=(timesteps, input_dim), batch_size=1, dtype=tf.float32, name="x_input")
    # reset_input = tf.keras.Input(shape=(), batch_size=1, dtype=tf.float32, name="reset_input")

    # # Functional wrapper
    # output = model([x_input, reset_input])
    # export_model = tf.keras.Model(inputs=[x_input, reset_input], outputs=output)

    # # Build weights (needed before conversion)
    # # FIX: wrap in a list
    # _ = export_model([tf.random.uniform([1, 1, input_dim]), tf.constant([0.0], dtype=tf.float32)])

        # --------- QUANTIZATION: INT16 ACTIVATIONS + INT8 WEIGHTS ---------
    def representative_dataset():
        for i in range(100):
            x = np.random.uniform(-32768 / 2**8, 32767 / 2**8, size=(1, 1, input_dim)).astype(np.float32)
            reset_value = 0.0 if i < 80 else 1.0  # 80% no-reset, 20% reset
            reset = np.array([reset_value], dtype=np.float32)
            yield {"x_input": x, "reset_input": reset}


    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if dtype=="int8":
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS
            ]  # enable TensorFlow ops.
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.representative_dataset = representative_dataset
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
        converter.representative_dataset = representative_dataset
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

    # Convert and save
    tflite_model = converter.convert()
    with open(path_tflite, "wb") as f:
        f.write(tflite_model)

    print(f"✅ Saved model: {path_tflite}")

if __name__ == "__main__":
    # --------- BUILD MODEL ---------
    BATCH_SIZE = 1
    TIME_STEPS = 1
    INPUT_DIM = 257

    import numpy as np
    import tensorflow as tf
    model = StatefulLSTMModel_VAD(INPUT_DIM)
    path_tflite = "stateful_lstm_16x8_int_reset.tflite"
    # Convert the model to TFLite format
    tflite_conversion(
        model,
        timesteps=1,
        input_dim=INPUT_DIM,
        path_tflite=path_tflite)
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=path_tflite)
    interpreter.allocate_tensors()

    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Print input/output info (optional debug)
    for i, d in enumerate(input_details):
        print(f"Input {i}: name={d['name']} shape={d['shape']} dtype={d['dtype']}")

    print(f"Output: name={output_details[0]['name']} shape={output_details[0]['shape']} dtype={output_details[0]['dtype']}")

    def run_step(x_input_fp32, reset_flag):
        # Quantize input to int16
        x_input_scale, x_input_zero = input_details[1]['quantization']
        x_input_int16 = np.clip(x_input_fp32 / x_input_scale + x_input_zero, -32768, 32767).astype(np.int16)

        # Prepare reset input (already int16)

        if reset_flag==1:
            import pdb; pdb.set_trace()
        reset_scale, reset_input_zero = input_details[0]['quantization']
        reset_flag = np.clip(reset_flag / reset_scale + reset_input_zero, -32768, 32767).astype(np.int16)
        reset_input = np.array([reset_flag], dtype=np.int16)

        # Set tensors
        interpreter.set_tensor(input_details[1]['index'], x_input_int16)
        interpreter.set_tensor(input_details[0]['index'], reset_input)

        # Run inference
        interpreter.invoke()

        # Get output (int16)
        output_int16 = interpreter.get_tensor(output_details[0]['index'])
        y_scale, y_zero = output_details[0]['quantization']
        y_fp32 = (output_int16.astype(np.float32) - y_zero) * y_scale

        return y_fp32

    # Run 3 steps with reset = 0, then reset = 1
    for i in range(5):
        x = np.random.uniform(-1, 1, size=(1, 1, INPUT_DIM)).astype(np.float32)
        reset = 1 if i == 3 else 0
        y = run_step(x, reset_flag=reset)
        print(f"Step {i}, Reset={reset}, Output: {y}")

    BATCH_SIZE_TRAIN = 3
    model_train = StatefulLSTMModel_VAD(INPUT_DIM)

    x_input = tf.random.uniform([BATCH_SIZE_TRAIN, 1, INPUT_DIM])
    reset_input = tf.constant([0] * BATCH_SIZE_TRAIN, dtype=tf.int16)

    # Build and initialize variables
    _ = model_train([x_input, reset_input])

    # Save weights
    model_train.save_weights("trained_model_weights.h5")
    print("✅ Weights saved from batch_size=3 model.")


    BATCH_SIZE_INFER = 1
    model_infer = StatefulLSTMModel_VAD(INPUT_DIM)

    # Build with dummy input to trigger variable creation
    x_input = tf.random.uniform([BATCH_SIZE_INFER, 1, INPUT_DIM])
    reset_input = tf.constant([0], dtype=tf.float32)
    _ = model_infer([x_input, reset_input])

    # Load weights (shared weights like Conv2D and LSTM — not state variables)
    model_infer.load_weights("trained_model_weights.h5", by_name=True, skip_mismatch=True)
    print("✅ Weights loaded into batch_size=1 model.")
