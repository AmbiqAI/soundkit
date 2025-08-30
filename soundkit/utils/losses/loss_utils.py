import tensorflow as tf
from ..tf_complex_utils import polar_to_complex, complex_diff_square
from soundkit.utils.tf_basic_math import tf_log10_eps
import numpy as np
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
        if tf.as_dtype(y_true.dtype).is_complex:
            mag_y_pred = tf_log10_eps(tf.abs(y_pred))
            mag_y_true = tf_log10_eps(tf.abs(y_true))
            loss = tf.reduce_sum(
                tf.sqrt((mag_y_pred - mag_y_true)**2 + 1e-12)
            ) / tf.cast(steps, tf.float32)
            

            # Normalize by the magnitude of the true signal
            # loss = tf.reduce_sum(
            #     complex_diff_square(y_pred, y_true)
            # ) / tf.cast(steps, tf.float32)
           
        else:
            loss = tf.reduce_sum(tf.square(y_pred - y_true)) / tf.cast(steps, tf.float32)

        return loss


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
        self.weights = self.get_energy_based_weights()

    def get_energy_based_weights(self, fft_bins=257):
        weights = np.ones(fft_bins)
        
        # Define bin ranges based on 16kHz, 512-point STFT (31.25 Hz/bin)
        weights[0:8] = 1.0        # 0–250 Hz
        weights[8:16] = 0.8       # 250–500 Hz
        weights[16:32] = 1.0      # 500–1k
        weights[32:64] = 1.3      # 1–2k
        weights[64:128] = 2.0     # 2–4k
        weights[128:192] = 20.0    # 4–6k
        weights[192:257] = 20.0    # 6–8k

        # Normalize to mean 1
        weights /= np.mean(weights)
        weights = np.reshape(weights, (1, 1, -1))  # Reshape for broadcasting
        return tf.constant(weights, dtype=tf.float32)


    def call(self, x, y):
        """
        Compute compressed MSE over all elements.

        Args:
            x: Predicted tensor (real or complex)
            y: Ground truth tensor (real or complex)

        Returns:
            Scalar tensor loss
        """
        steps = tf.shape(x)[0] * tf.shape(x)[1]
        eps_cmlx=self.eps + 0j
        mag_y = tf.abs(y)
        mag_x = tf.abs(x)

        mask = mag_y >= self.eps

        y_safe = tf.where(mask, y, y + eps_cmlx)
        x_safe = tf.where(mask, x, x + eps_cmlx)


        if tf.as_dtype(x.dtype).is_complex:

            phase_x = tf.math.angle(x)
            
            phase_y = tf.math.angle(y_safe)

            x_comp_amp = tf.abs(x_safe) ** self.exp
            y_comp_amp = tf.abs(y_safe) ** self.exp

            x_comp = polar_to_complex(x_comp_amp, phase_x)
            y_comp = polar_to_complex(y_comp_amp, phase_y)

            # weights = 1.0 / tf.maximum(
            #     tf.reduce_sum(x_comp_amp, axis=[0, 1]),
            #     self.eps)

            # # make dc component less important
            # weights = tf.concat([[weights[0] / 20.0], weights[1:]], axis=0)

            # weights_sum = tf.reduce_sum(weights)

            # equal_weights = tf.ones_like(weights) / tf.reduce_sum(tf.ones_like(weights))

            # weights = tf.where(
            #     tf.math.is_finite(weights_sum),
            #     weights / weights_sum,
            #     equal_weights
            # )
            if 1:
                # # err = complex_diff_square(x_comp, y_comp) * weights
                freq_bins = tf.shape(x_comp)[-1]
                half = freq_bins // 2
                prob = tf.random.uniform([], 0.0, 1.0)

                # Binary mask: [0, 1, 1, 1, ...] for high freq or [1, 1, 0, 0, ...] for low freq
                mask_high = tf.concat([tf.zeros(half), tf.ones(freq_bins - half)], axis=0)
                mask_low = tf.concat([tf.ones(half), tf.zeros(freq_bins - half)], axis=0)
                freq_mask = tf.cond(prob < 1, lambda: mask_high, lambda: mask_low)
                freq_mask = tf.reshape(freq_mask, [1, 1, -1])  # For broadcasting over [B, T, F]

                # Compute masked complex difference loss
                weights = tf.reduce_mean(x_comp_amp**2, axis=[0, 1])
                weights = 1 / tf.maximum(
                    weights,
                    self.eps
                )
                weights = tf.concat([[weights[1] / 20.0], weights[1:]], axis=0)
                weights = weights / tf.reduce_sum(weights)
                # import matplotlib.pyplot as plt
                # plt.plot(weights.numpy())
                # plt.title("Frequency Weights")
                # plt.xlabel("Frequency Bin")
                # plt.ylabel("Weight")
                # plt.grid()
                # plt.show()
                # import pdb; pdb.set_trace()
                weights = tf.reshape(weights, [1, 1, -1])  # For broadcasting over [B, T, F]

                err = complex_diff_square(x_comp, y_comp)

                loss = tf.reduce_sum(err) / tf.cast(steps, tf.float32)
            else:
                freq_bins = tf.shape(x_comp)[-1]
                half = freq_bins // 2
                prob = tf.random.uniform([], 0.0, 1.0)

                # Binary mask: [0, 1, 1, 1, ...] for high freq or [1, 1, 0, 0, ...] for low freq
                mask_high = tf.concat([tf.zeros(half), tf.ones(freq_bins - half)], axis=0)
                mask_low = tf.concat([tf.ones(half), tf.zeros(freq_bins - half)], axis=0)
                freq_mask = tf.cond(
                    prob < 1.0,
                    lambda: mask_high,
                    lambda: mask_low)
                freq_mask = tf.reshape(freq_mask, [1, 1, -1])  # For broadcasting over [B, T, F]

                err = (tf_log10_eps(tf.abs(x_safe))-tf_log10_eps(tf.abs(y_safe)))**2
                loss = tf.reduce_sum(err * freq_mask) / tf.cast(steps, tf.float32)           
        else:
            x_comp = (mag_x ** self.exp)
            y_comp = (mag_y ** self.exp)
            loss = tf.reduce_sum(tf.square(x_comp - y_comp)) / tf.cast(steps, tf.float32)
        return loss

