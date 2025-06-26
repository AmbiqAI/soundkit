import tensorflow as tf
import numpy as np

# --------- MODEL DEFINITION ---------
class StatefulLSTMModel(tf.keras.Model):
    def __init__(self, input_dim, lstm_units, batch_size=1):
        super(StatefulLSTMModel, self).__init__()
        self.lstm_units = lstm_units
        self.batch_size = batch_size


        self.conv2d = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(1, 4),
            activation='relu',
            padding='valid',
        )

        self.lstm = tf.keras.layers.LSTM(
            lstm_units,
            return_sequences=False,
            return_state=True,
            unroll=True,
        )

        self.input_state=None
        #
        self.h_state = None
        self.c_state = None
        # self.input_state=tf.Variable(tf.zeros([batch_size, 2, 4]), trainable=False, name="input_state")
        # #
        # self.h_state = tf.Variable(tf.zeros([batch_size, lstm_units]), trainable=False, name="h_state")
        # self.c_state = tf.Variable(tf.zeros([batch_size, lstm_units]), trainable=False, name="c_state")
    
    def build(self, input_shape):
        batch_size = input_shape[0][0]

        self.input_state = self.add_weight(
            shape=(batch_size, 2, input_shape[0][2]),
            trainable=False,
            name="input_state",
            initializer="zeros"
        )
        self.h_state = self.add_weight(
            shape=(batch_size, self.lstm_units),
            trainable=False,
            name="h_state",
            initializer="zeros"
        )
        self.c_state = self.add_weight(
            shape=(batch_size, self.lstm_units),
            trainable=False,
            name="c_state",
            initializer="zeros"
        )
    @tf.function
    def reset(self):
        """Reset the internal states of the LSTM and input state."""
        self.input_state.assign(tf.zeros_like(self.input_state))
        self.h_state.assign(tf.zeros_like(self.h_state))
        self.c_state.assign(tf.zeros_like(self.c_state))

    def call(self, inputs, training=False):
        x_input, reset_input = inputs  # reset_input: [B, 1] int16

        # Determine whether to reset
        should_reset = tf.cast(tf.math.abs(reset_input), tf.bool)[0]

        # Execute the reset inside a control flow block
        tf.cond(
            pred=should_reset,
            true_fn=lambda: self.reset(),
            false_fn=lambda: None
        )
        x_input = tf.concat([self.input_state, x_input], axis=1)  # Concatenate input with state
        input_state = tf.identity(x_input[:, -2:, :])  # Keep the last two time steps for input state
        
        
        x_input = tf.expand_dims(x_input, axis=-1)  # Add channel dimension for Conv2D
        
        
        x_input = self.conv2d(x_input)  # Apply Conv2D layer
        x_input = tf.squeeze(x_input, axis=-2)
        # Proceed with normal LSTM forward pass
        output, h_new, c_new = self.lstm(x_input, initial_state=[self.h_state, self.c_state])
        self.input_state.assign(input_state)  # Update input state
        self.h_state.assign(h_new)
        self.c_state.assign(c_new)

        return output


# --------- BUILD MODEL ---------
BATCH_SIZE = 1
TIME_STEPS = 1
INPUT_DIM = 4
LSTM_UNITS = 8

model = StatefulLSTMModel(INPUT_DIM, LSTM_UNITS, BATCH_SIZE)

# Inputs
x_input = tf.keras.Input(shape=(TIME_STEPS, INPUT_DIM), batch_size=BATCH_SIZE, name="x_input")
reset_input = tf.keras.Input(shape=(), batch_size=BATCH_SIZE, dtype=tf.int16, name="reset_input")

# Functional wrapper
output = model([x_input, reset_input])
export_model = tf.keras.Model(inputs=[x_input, reset_input], outputs=output)

# Build weights (needed before conversion)
# FIX: wrap in a list
_ = export_model([tf.random.uniform([1, 1, INPUT_DIM]), tf.constant([0], dtype=tf.int16)])

# --------- QUANTIZATION: INT16 ACTIVATIONS + INT8 WEIGHTS ---------
def representative_dataset():
    for i in range(100):
        x = np.random.uniform(-1, 1, size=(1, 1, INPUT_DIM)).astype(np.float32)
        reset_value = 0 if i < 80 else 1  # 80% no-reset, 20% reset
        reset = np.array([reset_value], dtype=np.int16)
        yield {"x_input": x, "reset_input": reset}


converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
# Force int16 input/output
converter.inference_input_type = tf.int16
converter.inference_output_type = tf.int16

# Convert and save
tflite_model = converter.convert()
with open("stateful_lstm_16x8_int_reset.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Saved model: stateful_lstm_16x8_int_reset.tflite")


import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="stateful_lstm_16x8_int_reset.tflite")
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
    x = np.random.uniform(-1, 1, size=(1, 1, 4)).astype(np.float32)
    reset = 1 if i == 3 else 0
    y = run_step(x, reset_flag=reset)
    print(f"Step {i}, Reset={reset}, Output: {y}")

BATCH_SIZE_TRAIN = 3
model_train = StatefulLSTMModel(INPUT_DIM, LSTM_UNITS, batch_size=BATCH_SIZE_TRAIN)

x_input = tf.random.uniform([BATCH_SIZE_TRAIN, 1, INPUT_DIM])
reset_input = tf.constant([0] * BATCH_SIZE_TRAIN, dtype=tf.int16)

# Build and initialize variables
_ = model_train([x_input, reset_input])

# Save weights
model_train.save_weights("trained_model_weights.h5")
print("✅ Weights saved from batch_size=3 model.")


BATCH_SIZE_INFER = 1
model_infer = StatefulLSTMModel(INPUT_DIM, LSTM_UNITS, batch_size=BATCH_SIZE_INFER)

# Build with dummy input to trigger variable creation
x_input = tf.random.uniform([BATCH_SIZE_INFER, 1, INPUT_DIM])
reset_input = tf.constant([0], dtype=tf.int16)
_ = model_infer([x_input, reset_input])

# Load weights (shared weights like Conv2D and LSTM — not state variables)
model_infer.load_weights("trained_model_weights.h5", by_name=True, skip_mismatch=True)
print("✅ Weights loaded into batch_size=1 model.")
