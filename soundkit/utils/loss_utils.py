import tensorflow as tf
from typing import Callable

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


class FramewiseMSE(tf.keras.losses.Loss):
    """
    Framewise Mean Squared Error computed across [B, T, ...] flattened.
    """
    def __init__(self, name="framewise_mse"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        """
        Compute mean squared error over all time-frequency points.

        Args:
            y_true: Ground truth tensor
            y_pred: Predicted tensor

        Returns:
            Scalar tensor representing the MSE
        """
        steps = tf.shape(y_true)[0] * tf.shape(y_true)[1]
        return tf.reduce_sum(tf.square(y_pred - y_true)) / tf.cast(steps, tf.float32)


class FramewiseMAE(tf.keras.losses.Loss):
    """
    Framewise Mean Absolute Error computed across [B, T, ...] flattened.
    """
    def __init__(self, name="framewise_mae"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        """
        Compute mean absolute error over all time-frequency points.

        Args:
            y_true: Ground truth tensor
            y_pred: Predicted tensor

        Returns:
            Scalar tensor representing the MAE
        """
        steps = tf.shape(y_true)[0] * tf.shape(y_true)[1]
        return tf.reduce_sum(tf.abs(y_pred - y_true)) / tf.cast(steps, tf.float32)


class CompressedMSE(tf.keras.losses.Loss):
    """
    Compressed Mean Squared Error:
    Applies a power-law compression to magnitude before MSE.

    For real tensors:
        cMSE = MSE(sign(x) * |x|^exp, sign(y) * |y|^exp)

    For complex tensors:
        cMSE = MSE(|x|^exp ∠ angle(x), |y|^exp ∠ angle(y))
    """
    def __init__(self, exp: float = 0.6, name="compressed_mse"):
        """
        Args:
            exp (float): Compression exponent, typically 0.3–0.6
            name (str): Optional loss name
        """
        super().__init__(name=name)
        self.exp = exp

    def call(self, x, y):
        """
        Compute compressed MSE over all elements.

        Args:
            x: Predicted tensor (real or complex)
            y: Ground truth tensor (real or complex)

        Returns:
            Scalar tensor loss
        """
        mag_x = tf.abs(x)
        mag_y = tf.abs(y)

        if tf.as_dtype(x.dtype).is_complex:

            phase_x = tf.math.angle(x)
            phase_y = tf.math.angle(y)
            x_comp = polar_to_complex(mag_x ** self.exp, phase_x)
            y_comp = polar_to_complex(mag_y ** self.exp, phase_y)
        else:
            x_comp = mag_x ** self.exp * tf.sign(x)
            y_comp = mag_y ** self.exp * tf.sign(y)

        steps = tf.shape(x)[0] * tf.shape(x)[1]
        loss = tf.reduce_sum(tf.square(x_comp - y_comp)) / tf.cast(steps, tf.float32)
        return loss


class LossFactory:
    """
    Factory class to construct and invoke consistent loss functions,
    including MSE, MAE, and custom compressed MSE.
    """

    def __init__(self, loss_type: str, **kwargs) -> None:
        """
        Initialize the loss function by type.

        Args:
            loss_type (str): One of "mse", "mae", "cmse"
            kwargs: Extra keyword arguments (e.g., exp for CompressedMSE)
        """
        self.loss_type = loss_type
        self.loss_fn = self._get_loss_function(loss_type, **kwargs)

    def _get_loss_function(self, loss_type: str, **kwargs) -> Callable:
        """
        Map loss type string to a loss function.

        Args:
            loss_type: Key indicating loss type
            kwargs: Passed to loss constructor if applicable

        Returns:
            Callable loss function
        """
        loss_map = {
            "mse": FramewiseMSE(),
            "mae": FramewiseMAE(),
            "compressed_mse": CompressedMSE(**kwargs),
        }

        if loss_type not in loss_map:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        return loss_map[loss_type]

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Call the selected loss function.

        Args:
            y_true: Ground truth tensor
            y_pred: Predicted tensor

        Returns:
            Scalar loss value
        """
        return self.loss_fn(y_true, y_pred)
