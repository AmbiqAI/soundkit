import tensorflow as tf

def polar_to_complex(magnitude: tf.Tensor, phase: tf.Tensor) -> tf.Tensor:
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

def complex_to_polar(
        complex_tensor: tf.Tensor,
        eps: float = 1e-12) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Convert complex tensor to polar coordinates (magnitude, phase).
    
    Args:
        complex_tensor: Complex-valued tensor
        
    Returns:
        Tuple of (magnitude, phase) tensors
    """
    # why adding 1e-12 here? to avoid sqrt(0) which may lead to NaN in backprop
    magnitude = tf.sqrt(
        tf.math.real(complex_tensor)**2 + tf.math.imag(complex_tensor)**2 + eps
        )

    # why adding 1e-12 here? 0+0j is undefined in angle
    phase = tf.math.angle(complex_tensor + eps)
    return magnitude, phase

def complex_diff_square(
        x: tf.Tensor,
        y: tf.Tensor) -> tf.Tensor:
    """
    Compute the squared difference between two complex tensors.
    
    Args:
        x: First complex tensor
        y: Second complex tensor
        
    Returns:
        Tensor of absolute differences
    """
    real_err = tf.math.real(x) - tf.math.real(y)
    imag_err = tf.math.imag(x) - tf.math.imag(y)
    error = real_err**2 + imag_err**2
    return  error


def complex_to_realarray(complex_tensor: tf.Tensor) -> tf.Tensor:
    """
    Convert complex tensor to real-valued array.
    
    Args:
        complex_tensor: Complex-valued tensor
        
    Returns:
        Real-valued tensor with shape [B, T, 2*F]
    """
    real_part = tf.math.real(complex_tensor)
    imag_part = tf.math.imag(complex_tensor)
    return tf.stack(
        [real_part, imag_part],
        axis=-1)

def realarray_to_complex(real_array: tf.Tensor) -> tf.Tensor:
    """
    Convert real-valued array to complex tensor.
    
    Args:
        real_array: Real-valued tensor with shape [B, T, 2*F]
        
    Returns:
        Complex-valued tensor
    """
    real_part = real_array[..., 0]
    imag_part = real_array[..., 1]
    return tf.complex(real_part, imag_part)