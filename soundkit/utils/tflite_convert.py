""" Convert the model to tflite format """
import os
import tempfile
import shutil
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
    # Use SavedModel path to work around a TFLite from_keras_model bug that
    # corrupts weights when Sequential layers with custom sublayers coexist
    # with resource-variable assign ops (e.g. stateful CRNN models).
    saved_model_dir = tempfile.mkdtemp()
    model.save(saved_model_dir)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_enable_resource_variables = True
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

    result = converter.convert()
    shutil.rmtree(saved_model_dir, ignore_errors=True)
    return result

def test_and_export_c(tflite_model_bytes, dtype="int8", output_dir="tflite_data"):
    """
    Runs a 2-frame stateful test and exports input/output to C files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize Interpreter (disable XNNPACK to avoid weight corruption
    #    with large Conv2D kernels)
    interpreter = tf.lite.Interpreter(
        model_content=tflite_model_bytes,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # 2. Prepare Input
    # Generate random float data first
    input_shape = input_details['shape']
    test_input_float = np.random.uniform(-1, 1, input_shape).astype(np.float32) * 5

    # Quantize input for the C array if necessary
    if dtype != "float32":
        in_scale, in_zp = input_details['quantization']
        input_data = ((test_input_float / in_scale) + in_zp).astype(input_details['dtype'])
    else:
        input_data = test_input_float

    # 3. Helper to write C files
    def write_c_array(name, data, filename):
        c_type = "int8_t" if dtype == "int8" else "int16_t" if dtype == "int16" else "float"
        flat = data.flatten()
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(f'#include <stdint.h>\n\nconst {c_type} {name}[] = {{\n    ')
            for i, val in enumerate(flat):
                f.write(f"{int(val)}, " if "int" in c_type else f"{val:.8f}f, ")
                if (i + 1) % 12 == 0: f.write("\n    ")
            f.write(f"\n}};\nconst int {name}_len = {len(flat)};\n")
        print(f"💾 Exported: {filename}")

    # 4. Run 2 Frames and Capture
    results = []
    interpreter.set_tensor(input_details['index'], input_data)
    
    print(f"\n--- Running Stateful Test ({dtype}) ---")
    for i in range(2):
        interpreter.invoke()
        out = interpreter.get_tensor(output_details['index'])
        results.append(out.copy())
        print(f"Frame {i+1} - Mean: {np.mean(out):.4f}")

    # 5. Export to C
    write_c_array("input_frame", input_data, f"input_{dtype}.c")
    write_c_array("output_frame1", results[0], f"output_f1_{dtype}.c")
    write_c_array("output_frame2", results[1], f"output_f2_{dtype}.c")

    # 6. Verification Logic
    diff = np.abs(results[0] - results[1]).sum()
    if diff > 1e-6:
        print(f"✅ Success: Statefulness detected (Diff: {diff:.6f})")
    else:
        print("⚠️ Warning: Frames are identical. Check stateful settings.")

    return results

def tflite_convert(
        model: tf.keras.Model,
        dtype: str = "int8",
        path_tflite: str = "./tflite/nnse.tflite",
        nbits: int = 16,
        qbits: int = 8,
        data_calibration: np.ndarray | None = None):
    """tflite converter"""

    os.makedirs('tflite', exist_ok=True)

    min_val = -2**(nbits-1) / 2**qbits
    max_val = (2**(nbits-1) - 1) / 2**qbits

    def dataset_example():
        shapes = model._feed_input_shapes
        shape_inputs = shapes[0]
        timesteps = shape_inputs[1]

        if data_calibration is None:
            
            for _ in range(100):
                x = np.random.uniform(
                    min_val, max_val,
                    size=shape_inputs).astype(np.float32)
                yield {"x_input": x}

        # Load a few real samples from your validation set
        else:
            max_calib_frames = 300
            count = 0
            for sequence in data_calibration:
                # Reset all non-trainable variables (states) to zero
                for var in model.variables:
                    if not var.trainable:
                        var.assign(tf.zeros_like(var))

                for t in range(sequence.shape[0] // timesteps):
                    start = t * timesteps
                    end = (t + 1) * timesteps
                    frame = sequence[start:end].reshape(shape_inputs).astype(np.float32)
                    yield {"x_input": frame}
                    # count += 1
                    # if count >= max_calib_frames:
                    #     return

    model.summary()

    # Check if model states are zeros, reset if not
    print("\n--- Model State Check ---")
    all_zero = True
    for var in model.variables:
        if not var.trainable:
            is_zero = np.allclose(var.numpy(), 0.0)
            norm = np.linalg.norm(var.numpy())
            print(f"  {var.name:40s} | shape={str(var.shape):20s} | zero={is_zero} | norm={norm:.6f}")
            if not is_zero:
                all_zero = False
    if not all_zero:
        print("⚠️  Non-zero states detected, resetting to zero...")
        for layer in model.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states(zero_state=True)
                break
    else:
        print("✅ All states are zero.")
    print("-------------------------\n")

    net_tflite = convert_model(
        model,
        dataset_example,
        dtype=dtype)

    test_and_export_c(net_tflite, dtype=dtype)

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

    outputs= model(inputs_feat, training=False)

    model_wrap = tf.keras.Model(
        inputs=[
            inputs_feat,
            ],
        outputs=[
            outputs,
            ])

    return model_wrap
