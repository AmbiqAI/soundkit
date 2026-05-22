"""
Copy model weights from one model to another in TensorFlow.
"""
import tensorflow as tf
import re
def copy_model_weights(
        model_dst: tf.keras.Model,
        model_src: tf.keras.Model
    ) -> None:
    """Copy model weights from source to destination model."""
    for s, d in zip(model_src.variables, model_dst.variables):
        print(f"{s.name} -> {d.name }")
        if re.search("moving_variance", s.name):
            print(s.name)
            print(f"min = {tf.reduce_min(s)}, max = {tf.reduce_max(s)}")
        elif re.search("gamma", s.name):
            print(s.name)
            print(f"min = {tf.reduce_min(s)}, max = {tf.reduce_max(s)}")
        if s.shape != d.shape:
            print(f"Shape mismatch for {s.name} and {d.name}: {s.shape} vs {d.shape}")
        else:
            d.assign(s)