import tensorflow as tf

from soundkit.utils.tf_stft import tf_stft, tf_istft

class MultiResolutionSTFTLossFromSTFT(tf.keras.losses.Loss):
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
        if stft_configs is None:
            stft_configs = [
                {"n_fft": 256, "hop_length": 80, "win_length": 160},    # 5ms
                {"n_fft": 512, "hop_length": 160, "win_length": 320},   # 10ms
                {"n_fft": 1024, "hop_length": 320, "win_length": 640},  # 20ms
                {"n_fft": 2048, "hop_length": 640, "win_length": 1280}, # 40ms
            ]
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


    def stft_mag(self, waveform, n_fft, hop_length, win_length):
        """
        Compute magnitude STFT for a batch of waveforms
        """
        return tf.abs(tf_stft(
            waveform,
            frame_length=win_length,
            frame_step=hop_length,
            fft_length=n_fft,
        ))

    def call(self, y_true, y_pred):
        """
        y_true, y_pred: complex STFTs with shape (B, T, F)
        """
        from ..plot_api import plot_spectrograms
        wav_true = self.istft_batch(y_true)
        wav_pred = self.istft_batch(y_pred)
        losses = []
        for cfg in self.stft_configs:
            mag_true = self.stft_mag(wav_true, **cfg) + self.eps
            mag_pred = self.stft_mag(wav_pred, **cfg) + self.eps
            
            # plot_spectrograms(
            #         images=[pspec_s.T, pspec_sn.T],
            #         titles=["clean logspec", "noisy logspec"],
            #         vmin_vmax=[(-80, 10), (-80, 10)],
            #         show_colorbar=True,
            #         show_fig=True       # set False if only saving
            #     )
            
            
            steps = tf.shape(mag_pred)[0] * tf.shape(mag_pred)[1]
            loss = tf.reduce_sum((mag_true**self.exp - mag_pred**self.exp)**2) / tf.cast(steps, tf.float32)    
            losses.append(loss)

        return tf.reduce_mean(losses)#