import tensorflow as tf

from soundkit.utils.tf_stft import (
    tf_istft,
)
from ..plot_api import plot_spectrograms
class SI_SDR(tf.keras.losses.Loss):
    def __init__(self,
                 stft_configs=None,
                 fft_size=256,
                 frame_size=256,
                 hop_length=128,
                 exp = 0.6,
                 eps=1e-12,
                 name="multi_resolution_stft_loss",
                 **kwargs):
        super().__init__(name=name)

        self.exp = exp
        self.eps = eps
        self.stft_configs = stft_configs
        self.fft_size = fft_size
        self.frame_size = frame_size
        self.hop_length = hop_length

    def istft_batch(self, stft_batch):
        """
        Apply inverse STFT on batch of shape (B, T, F)
        """

        return tf_istft(
                    stft_batch,
                    frame_length=self.frame_size,
                    frame_step=self.hop_length,
                    fft_length=self.fft_size,
                )

    def call(self, y_true, y_pred):
        """
        Computes SI-SDR loss between predicted and reference waveforms.
        
        Args:
            y_true: Ground truth waveform. Shape (B, T)
            y_pred: Predicted waveform. Shape (B, T)
        
        Returns:
            Negative SI-SDR (to minimize as loss)
        """
            
        wav_true = self.istft_batch(y_true)
        wav_pred = self.istft_batch(y_pred)

        # Ensure zero-mean

        wav_true = wav_true - tf.reduce_mean(wav_true, axis=1, keepdims=True)
        wav_pred = wav_pred - tf.reduce_mean(wav_pred, axis=1, keepdims=True)

        # Compute projection of y_pred onto y_true
        dot = tf.reduce_sum(wav_true * wav_pred, axis=1, keepdims=True)  # (B, 1)
        norm_sq = tf.reduce_sum(wav_true ** 2, axis=1, keepdims=True) + self.eps  # (B, 1)

        scale = dot / norm_sq
        proj = scale * wav_true  # (B, T)

        # Compute SI-SDR
        noise = wav_pred - proj
        ratio = tf.reduce_sum(proj ** 2, axis=1) / (tf.reduce_sum(noise ** 2, axis=1) + self.eps)
        si_sdr = 10 * tf.math.log(ratio + self.eps) / tf.math.log(tf.constant(10., dtype=ratio.dtype))

        # Return as loss (negative SI-SDR)
        return -tf.reduce_mean(si_sdr)
    