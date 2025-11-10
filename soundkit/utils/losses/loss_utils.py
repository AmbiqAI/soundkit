import tensorflow as tf
from soundkit.utils.tf_complex_utils import (
    complex_to_realarray,
    polar_to_complex,
    complex_angle,
    complex_magnitude
)
class FramewiseMSE(tf.keras.losses.Loss):
    """
    Framewise Mean Squared Error computed across [B, T, ...] flattened.
    """
    def __init__(self, name="framewise_mse", **kwargs):
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
    def __init__(self, name="framewise_mae", **kwargs):
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
    def __init__(
            self,
            exp: float = 0.6,
            eps: float =1e-8,
            name="compressed_mse",
            **kwargs):
        """
        Args:
            exp (float): Compression exponent, typically 0.3–0.6
            eps (float): Small value the non-singularity
            name (str): Optional loss name
        """
        super().__init__(name=name)
        self.exp = float(exp)
        self.eps = float(eps)

    def call(self, x, y):
        """
        Compute compressed MSE over all elements.

        Args:
            x: Predicted tensor (real or complex)
            y: Ground truth tensor (real or complex)

        Returns:
            Scalar tensor loss
        """

        # if self.exp != 1.0:
        #     x = x + tf.complex(self.eps, 0.0)
        #     y = y + tf.complex(self.eps, 0.0)

        mag_x = tf.pow(
            complex_magnitude(x, self.eps),
            self.exp)
        mag_y = tf.pow(
            complex_magnitude(y, self.eps),
            self.exp)
        # # Check for NaNs immediately after magnitude compression
        # tf.print("mag_x_comp has NaNs:", tf.reduce_any(tf.math.is_nan(mag_x)))
        # tf.print("mag_y_comp has NaNs:", tf.reduce_any(tf.math.is_nan(mag_y)))

        x_comp = complex_to_realarray(
            polar_to_complex(mag_x, complex_angle(x, self.eps))
        )
        y_comp = complex_to_realarray(
            polar_to_complex(mag_y, complex_angle(y, self.eps))
        )

        # # Check for NaNs after reconstruction
        # tf.print("x_comp has NaNs:", tf.reduce_any(tf.math.is_nan(x_comp)))
        # tf.print("y_comp has NaNs:", tf.reduce_any(tf.math.is_nan(y_comp)))

        steps = tf.shape(x)[0] * tf.shape(x)[1]
        err = tf.abs(x_comp - y_comp)
        loss = tf.reduce_sum(tf.square(err)) / tf.cast(steps, tf.float32)
        return loss
    
