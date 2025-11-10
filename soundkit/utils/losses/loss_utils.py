"""Basic loss functions for soundkit.
Includes:
    - FramewiseMSE: Mean Squared Error computed frame-wise.
    - FramewiseMAE: Mean Absolute Error computed frame-wise.
    - LogFramewiseMSE: Logarithmic Framewise Mean Squared Error computed frame-wise.
    - CompressedMSE: Compressed Mean Squared Error with power-law compression.
    - SISDRLoss: Scale-Invariant Signal-to-Distortion Ratio Loss.
"""
import tensorflow as tf
from soundkit.utils.tf_complex_utils import (
    complex_to_realarray,
    polar_to_complex,
    complex_angle,
    complex_magnitude
)
from soundkit.utils.tf_basic_math import tf_log10_eps

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
        if y_true.dtype is not tf.complex64:
            raise ValueError("Input tensors must be of complex dtype.")
        if y_pred.dtype is not tf.complex64:
            raise ValueError("Input tensors must be of complex dtype.")
        steps = tf.shape(y_true)[0] * tf.shape(y_true)[1]
        err = tf.abs(y_pred - y_true)
        return tf.reduce_sum(tf.square(err)) / tf.cast(steps, tf.float32)

class LogFramewiseMSE(tf.keras.losses.Loss):
    """
    Log Framewise Mean Squared Error computed across [B, T, ...] flattened.
    """
    def __init__(self, eps=1e-8, name="log_framewise_mse", **kwargs):
        super().__init__(name=name)
        self.eps = eps

    def call(self, y_true, y_pred):
        """
        Compute log mean squared error over all time-frequency points.

        Args:
            y_true: Ground truth tensor
            y_pred: Predicted tensor

        Returns:
            Scalar tensor representing the log MSE
        """
        if y_true.dtype is not tf.complex64:
            raise ValueError("Input tensors must be of complex dtype.")
        if y_pred.dtype is not tf.complex64:
            raise ValueError("Input tensors must be of complex dtype.")
        steps = tf.shape(y_true)[0] * tf.shape(y_true)[1]

        pred_abs = 10 * tf_log10_eps(tf.abs(y_true)**2, self.eps)
        true_abs = 10 * tf_log10_eps(tf.abs(y_pred)**2, self.eps)

        err = tf.abs(pred_abs - true_abs)
    
        return tf.reduce_sum(tf.square(err)) / tf.cast(steps, tf.float32)

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
        if y_true.dtype is not tf.complex64:
            raise ValueError("Input tensors must be of complex dtype.")
        if y_pred.dtype is not tf.complex64:
            raise ValueError("Input tensors must be of complex dtype.")
        steps = tf.shape(y_true)[0] * tf.shape(y_true)[1]
        err = tf.abs(y_pred - y_true)
        return tf.reduce_sum(err) / tf.cast(steps, tf.float32)


class CompressedMSE(tf.keras.losses.Loss):
    """
    Compressed Mean Squared Error:
    Applies a power-law compression to magnitude before MSE.

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
        if x.dtype is not tf.complex64:
            raise ValueError("Input tensors must be of complex dtype.")
        if y.dtype is not tf.complex64:
            raise ValueError("Input tensors must be of complex dtype.")
        mag_x = tf.pow(
            complex_magnitude(x, self.eps),
            self.exp)
        mag_y = tf.pow(
            complex_magnitude(y, self.eps),
            self.exp)
        angle_x = complex_angle(x, self.eps)
        angle_y = complex_angle(y, self.eps)
        x_comp = complex_to_realarray(
            polar_to_complex(mag_x, angle_x)
        )
        y_comp = complex_to_realarray(
            polar_to_complex(mag_y, angle_y)
        )

        steps = tf.shape(x)[0] * tf.shape(x)[1]
        err = tf.abs(x_comp - y_comp)
        loss = tf.reduce_sum(tf.square(err)) / tf.cast(steps, tf.float32)
        return loss

class SISDRLoss(tf.keras.losses.Loss):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) Loss.

    Computes the negative SI-SDR (so lower = worse, higher = better).
    Can be used directly in model.compile(loss=SISDRLoss()).
    """

    def __init__(self, eps=1e-8, name="si_sdr_loss", **kwargs):
        super().__init__(name=name)
        self.eps = eps

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: Tensor of shape [batch, time], reference/clean signal.
            y_pred: Tensor of shape [batch, time], estimated signal.

        Returns:
            Scalar tensor: mean negative SI-SDR over batch.
        """
        # --- Input validation ---
        if y_true.dtype is not tf.float32:
            raise ValueError("Input tensors must be of float32 dtype.")
        if y_pred.dtype is not tf.float32:
            raise ValueError("Input tensors must be of float32 dtype.")

        # Ensure signals are real-valued
        # if not tf.reduce_all(tf.math.is_finite(y_true)):
        #     raise ValueError("y_true contains non-finite or complex values.")
        # if not tf.reduce_all(tf.math.is_finite(y_pred)):
        #     raise ValueError("y_pred contains non-finite or complex values.")

        if y_true.shape.rank != 2 or y_pred.shape.rank != 2:
            raise ValueError(
                f"Inputs must have shape [batch, time], got {y_true.shape} and {y_pred.shape}"
            )

        # --- Zero-mean normalization ---
        y_true -= tf.reduce_mean(y_true, axis=1, keepdims=True)
        y_pred -= tf.reduce_mean(y_pred, axis=1, keepdims=True)

        # --- Target projection (scale-invariant) ---
        dot = tf.reduce_sum(y_true * y_pred, axis=1, keepdims=True)
        ref_energy = tf.reduce_sum(y_true ** 2, axis=1, keepdims=True) + self.eps
        scaling = dot / ref_energy
        target = scaling * y_true

        # --- Compute residual noise ---
        noise = y_pred - target

        # --- Compute SI-SDR in dB ---
        ratio = tf.reduce_sum(target ** 2, axis=1) / (
            tf.reduce_sum(noise ** 2, axis=1) + self.eps
        )
        si_sdr = 10 * tf.math.log(ratio + self.eps) / tf.math.log(10.0)

        # Return negative mean (as loss to minimize)
        return -tf.reduce_mean(si_sdr)
