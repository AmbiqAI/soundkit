"""
Complex number utilities for TensorFlow.
"""
import tensorflow as tf

def complex_magnitude(
        complex_tensor: tf.Tensor,
        eps: float = 1e-8) -> tf.Tensor:
    """
    Compute the magnitude of a complex tensor.

    Args:
        complex_tensor: Complex-valued tensor
        eps: Small value to avoid sqrt(0)

    Returns:
        Real-valued tensor representing the magnitude of the complex tensor
    """
    # amp = tf.sqrt(tf.math.real(complex_tensor)**2 + tf.math.imag(complex_tensor)**2 + eps**2)
    # return amp
    mag = tf.abs(complex_tensor) 
    # Use max to ensure the value is at least eps for the power gradient
    return tf.maximum(mag, eps)
def complex_angle(
        complex_tensor: tf.Tensor,
        eps: float = 1e-8) -> tf.Tensor:
    """
    Compute the angle (phase) of a complex tensor.

    Args:
        complex_tensor: Complex-valued tensor
        eps: Small value to avoid division by zero

    Returns:
        Real-valued tensor representing the angle (phase) of the complex tensor
    """
    return tf.math.atan2(
        tf.math.imag(complex_tensor),
        tf.math.real(complex_tensor) + eps
    )

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

def get_compressed_complex(
        spec: tf.Tensor,
        exponent: float = 0.3,
        eps: float = 1e-7) -> tf.Tensor:
    """ Apply power-law compression to a complex spectrogram while preserving phase.
    
    Args:
        spec: Complex-valued spectrogram tensor of shape [B, T, F]
        exponent: Compression exponent (e.g., 0.3 for power-law compression)
        eps: Small value to avoid division by zero in phase calculation

    Returns:
        Complex tensor with power-law compressed magnitude and original phase
    """
    mag = tf.abs(spec)

    # Layer 1: Floor the magnitude for the power operation
    safe_mag = tf.maximum(mag, eps)

    # Layer 2: Calculate the scaling ratio (mag^-0.7)
    # This is where the 1/0 danger usually lives
    ratio = tf.pow(safe_mag, exponent - 1.0)

    # Layer 3: Hard-mask the ratio to 0 where the input was 0
    # This stops the "gradient noise" in silent regions
    ratio = tf.where(mag > 0, ratio, 0.0)

    # Apply to the original complex components
    # (spec already contains real + j*imag)
    return tf.complex(
        tf.math.real(spec) * ratio,
        tf.math.imag(spec) * ratio
    )

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
