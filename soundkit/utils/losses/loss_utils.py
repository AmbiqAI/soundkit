import tensorflow as tf
from ..tf_complex_utils import polar_to_complex

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
        mag_x = tf.abs(x) + self.eps
        mag_y = tf.abs(y) + self.eps

        if tf.as_dtype(x.dtype).is_complex:

            phase_x = tf.math.angle(x)
            phase_y = tf.math.angle(y)
            x_comp = polar_to_complex(mag_x ** self.exp, phase_x)
            y_comp = polar_to_complex(mag_y ** self.exp, phase_y)
        else:
            x_comp = (mag_x ** self.exp) * tf.sign(x)
            y_comp = (mag_y ** self.exp) * tf.sign(y)

        steps = tf.shape(x)[0] * tf.shape(x)[1]
        loss = tf.reduce_sum(tf.square(x_comp - y_comp)) / tf.cast(steps, tf.float32)
        return loss
