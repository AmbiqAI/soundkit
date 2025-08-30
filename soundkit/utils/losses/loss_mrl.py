import tensorflow as tf

from soundkit.utils.tf_stft import (
    tf_stft,
    tf_istft,
)
from soundkit.utils.tf_complex_utils import (
    polar_to_complex,
    complex_to_polar,
    complex_diff_square,
)
from ..plot_api import plot_spectrograms
class MultiResolutionSTFTLossFromSTFT(tf.keras.losses.Loss):
    def __init__(self,
                 stft_configs=None,
                 fft_size=256,
                 frame_size=256,
                 hop_length=128,
                 exp = 0.6,
                 eps=1e-12,
                 input_type="stft",
                 name="multi_resolution_stft_loss",
                 **kwargs):
        super().__init__(name=name)
        if stft_configs is None:
            stft_configs = [
                # {"n_fft": 128, "hop_length": 40, "win_length": 80},    # 2.5ms
                {"n_fft": 256, "hop_length": 80, "win_length": 160},    # 5ms
                {"n_fft": 512, "hop_length": 160, "win_length": 320},   # 10ms
                {"n_fft": 1024, "hop_length": 320, "win_length": 640},  # 20ms
                # {"n_fft": 2048, "hop_length": 640, "win_length": 1920}, # 40ms
            ]
        self.exp = exp
        self.eps = eps
        self.stft_configs = stft_configs
        self.fft_size = fft_size
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.input_type = input_type

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


    def stft_mag(self, waveform, n_fft, hop_length, win_length):
        """
        Compute magnitude STFT for a batch of waveforms
        """

        spec = tf_stft(
            waveform,
            frame_length=win_length,
            frame_step=hop_length,
            fft_length=n_fft,
        )
        mag, phase = complex_to_polar(spec, eps=self.eps)
        return mag, phase

    def stft(self, waveform, n_fft, hop_length, win_length):
        """
        Compute magnitude STFT for a batch of waveforms
        """
        return tf_stft(
            waveform,
            frame_length=win_length,
            frame_step=hop_length,
            fft_length=n_fft,
        )

    def call(self, y_true, y_pred):
        """
        y_true, y_pred: complex STFTs with shape (B, T, F)
        """

        wav_true = self.istft_batch(y_true)
        wav_pred = self.istft_batch(y_pred)
        losses = []
        for cfg in self.stft_configs:
            mag_true, phase_true = self.stft_mag(wav_true, **cfg)
            mag_pred, phase_pred = self.stft_mag(wav_pred, **cfg)

            mask = mag_pred >= self.eps
            mag_true_safe = tf.where(mask, mag_true, mag_true + self.eps)
            mag_pred_safe = tf.where(mask, mag_pred, mag_pred + self.eps)

            spec_compressed_true = polar_to_complex(
                mag_true_safe**self.exp,
                phase_true)

            spec_compressed_pred = polar_to_complex(
                mag_pred_safe**self.exp,
                phase_pred)
            # plot_spectrograms(
            #         images=[pspec_s.T, pspec_sn.T],
            #         titles=["clean logspec", "noisy logspec"],
            #         vmin_vmax=[(-80, 10), (-80, 10)],
            #         show_colorbar=True,
            #         show_fig=True       # set False if only saving
            #     )

            steps = tf.shape(mag_pred)[0] * tf.shape(mag_pred)[1]
            # # loss /=2
  
            freq_bins = tf.shape(mag_pred)[-1]
            half = freq_bins // 2
            prob = tf.random.uniform([], 0.0, 1.0)

            # Binary mask: [0, 1, 1, 1, ...] for high freq or [1, 1, 0, 0, ...] for low freq
            mask_high = tf.concat([tf.zeros(half), tf.ones(freq_bins - half)], axis=0)
            mask_low = tf.concat([tf.ones(half), tf.zeros(freq_bins - half)], axis=0)
            freq_mask = tf.cond(prob < 1.0, lambda: mask_high, lambda: mask_low)
            freq_mask = tf.reshape(freq_mask, [1, 1, -1])  # For broadcasting over [B, T, F]

            # Compute masked complex difference loss
            masked_diff = complex_diff_square(spec_compressed_true, spec_compressed_pred)
            masked_loss = tf.reduce_sum(masked_diff) / tf.cast(steps, tf.float32)

            losses.append(masked_loss)

        return tf.reduce_mean(losses)
    