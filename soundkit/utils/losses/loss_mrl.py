"""
Multi-Resolution STFT Loss Module
---------------------------------
Defines loss functions for audio enhancement using multi-scale Short-Time Fourier Transform (STFT).
Includes:
    - MultiResolutionSTFTLossFromSTFT: Computes loss by comparing magnitude STFTs at multiple resolutions.
Usage:
    Use as a custom loss in Keras models for speech/audio enhancement tasks.
"""

import tensorflow as tf

from soundkit.utils.tf_complex_utils import (
    polar_to_complex,
    complex_angle,
    complex_to_realarray,
    complex_magnitude,
)
from soundkit.utils.tf_stft import (
    tf_stft,
    tf_istft,
)

class MultiResolutionSTFTLossFromSTFT(tf.keras.losses.Loss):
    """
    Multi-Resolution STFT Loss for audio enhancement tasks.
    Computes loss by comparing magnitude STFTs at multiple resolutions.
    Inputs are complex STFTs; loss is calculated after ISTFT and multi-scale STFT.
    """
    def __init__(self,
                 stft_configs=None,
                 fft_size=512,
                 frame_size=480,
                 hop_length=128,
                 exp = 0.6,
                 eps=1e-8,
                 name="multi_resolution_stft_loss",
                 **kwargs):
        super().__init__(name=name)
        # Default STFT configs for multiple time-frequency resolutions
        if stft_configs is None:
            stft_configs = [
                {"n_fft": 256, "hop_length": 80, "win_length": 240},    # 5ms
                {"n_fft": 512, "hop_length": 160, "win_length": 320},   # 10ms
                {"n_fft": 1024, "hop_length": 320, "win_length": 640},  # 20ms
                {"n_fft": 2048, "hop_length": 640, "win_length": 1920}, # 40ms
            ]
        self.exp = exp  # Exponent for magnitude scaling
        self.eps = eps  # Small value to avoid log/zero issues
        self.stft_configs = stft_configs
        self.fft_size = fft_size
        self.frame_size = frame_size
        self.hop_length = hop_length

    def istft_batch(self, stft_batch):
        """
        Apply inverse STFT on batch of shape (B, T, F) to recover time-domain waveform.
        """
        return tf_istft(
            stft_batch,
            frame_length=self.frame_size,
            frame_step=self.hop_length,
            fft_length=self.fft_size,
        )


    def stft(self, waveform, n_fft, hop_length, win_length):
        """
        Compute magnitude STFT for a batch of waveforms at given resolution.
        """
        return tf_stft(
            waveform,
            frame_length=win_length,
            frame_step=hop_length,
            fft_length=n_fft,
        )

    def call(self, y_true, y_pred):
        """
        Compute multi-resolution STFT loss between predicted and true complex STFTs.

        Args:
            y_true (tf.Tensor): Ground truth complex STFTs, shape (B, T, F).
            y_pred (tf.Tensor): Predicted complex STFTs, shape (B, T, F).

        Returns:
            tf.Tensor: Scalar loss averaged over all resolutions.
        """
        # Convert STFTs back to time-domain waveforms
        wav_true = self.istft_batch(y_true)
        wav_pred = self.istft_batch(y_pred)
        losses = []
        # Compute loss at each STFT resolution
        for cfg in self.stft_configs:
            spec_true = self.stft(wav_true, **cfg)
            spec_pred = self.stft(wav_pred, **cfg)

            phase_true = complex_angle(spec_true, self.eps)
            phase_pred = complex_angle(spec_pred, self.eps)

            mag_true = complex_magnitude(spec_true, self.eps)
            mag_pred = complex_magnitude(spec_pred, self.eps)

            steps = tf.shape(mag_pred)[0] * tf.shape(mag_pred)[1]

            comp_true = polar_to_complex(tf.pow(mag_true, self.exp), phase_true)
            comp_pred = polar_to_complex(tf.pow(mag_pred, self.exp), phase_pred)
            abs_err = tf.abs(comp_pred - comp_true)

            loss = tf.reduce_sum(tf.square(abs_err)) / tf.cast(steps, tf.float32)
            losses.append(loss)

        # Average loss across all resolutions
        return tf.reduce_mean(losses)
