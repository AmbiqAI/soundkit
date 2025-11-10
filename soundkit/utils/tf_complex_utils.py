""""""
import tensorflow as tf

def complex_magnitude(
        complex_tensor: tf.Tensor,
        eps: float = 1e-8) -> tf.Tensor:
    """
    Compute the magnitude of a complex tensor.

    Args:
        complex_tensor: Complex-valued tensor

    Returns:
        Real-valued tensor representing the magnitude of the complex tensor
    """
    return tf.sqrt(tf.math.real(complex_tensor)**2 + tf.math.imag(complex_tensor)**2 + eps)

def complex_angle(
        complex_tensor: tf.Tensor,
        eps: float = 1e-8) -> tf.Tensor:
    """
    Compute the angle (phase) of a complex tensor.

    Args:
        complex_tensor: Complex-valued tensor

    Returns:
        Real-valued tensor representing the angle (phase) of the complex tensor
    """
    return tf.math.atan2(tf.math.imag(complex_tensor), tf.math.real(complex_tensor)+ eps)

def polar_to_complex(
        magnitude: tf.Tensor,
        phase: tf.Tensor) -> tf.Tensor:
    """
    Convert magnitude and phase to complex representation.
    
    Args:
        magnitude: Real-valued magnitude tensor
        phase: Real-valued phase tensor (radians)
        
    Returns:
        Complex tensor reconstructed from magnitude and phase
    """

    real = magnitude * tf.cos(phase)
    imag = magnitude * tf.sin(phase)

    return tf.complex(real, imag)

def complex_to_realarray(
        complex_tensor: tf.Tensor) -> tf.Tensor:
    """
    Convert complex tensor to real-valued array.
    
    Args:
        complex_tensor: Complex-valued tensor
        
    Returns:
        Real-valued tensor with shape [B, T, 2*F]
    """
    real_part = tf.math.real(complex_tensor)
    imag_part = tf.math.imag(complex_tensor)
    return tf.stack([real_part, imag_part], axis=-1)
