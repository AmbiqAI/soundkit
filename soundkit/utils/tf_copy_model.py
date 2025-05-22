"""
Copy model weights from one model to another in TensorFlow.
"""
import tensorflow as tf

def copy_model_weights(
        model_dst: tf.keras.Model,
        model_src: tf.keras.Model
    ) -> None:
    """Copy model weights from source to destination model."""
    for s, d in zip(model_src.trainable_variables, model_dst.trainable_variables):
        # if s.name != d.name:
        #     raise ValueError(f"Model weights do not match: {s.name} != {d.name}")
        # print(s.name, d.shape)
        # print(d.name, d.shape)
        
        d.assign(s)
        
    #     z = s.numpy().flatten()
    #     print(d.name, d.shape, z[1:5])
    # import pdb; pdb.set_trace()
        