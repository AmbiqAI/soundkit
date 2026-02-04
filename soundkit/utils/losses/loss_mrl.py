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
                 hop_length=160,
                 exp = 0.6,
                 eps=1e-8,
                 name="multi_resolution_stft_loss",
                 **kwargs):
        super().__init__(name=name)
        # Default STFT configs for multiple time-frequency resolutions
        if stft_configs is None:
            stft_configs = [
                {"n_fft": 256, "hop_length": 80, "win_length": 240},    # 5ms
                {"n_fft": 512, "hop_length": 160, "win_length": 480},   # 10ms
                {"n_fft": 1024, "hop_length": 320, "win_length": 960},  # 20ms
                {"n_fft": 2048, "hop_length": 640, "win_length": 1920}, # 40ms
            ]
        self.exp = exp  # Exponent for magnitude scaling
        self.eps = eps  # Small value to avoid log/zero issues
        self.stft_configs = stft_configs
        self.fft_size = fft_size
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.group_filter = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            trainable=False,
            use_bias=False,
            padding='SAME',
            kernel_initializer=tf.keras.initializers.Ones()
        )

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

    def _compute_binary_filled_vad(
        self,
        hop_size,
        y_true_wave,
        threshold=1e-4,
        target_ratio=0.1,
        fill_window=5): # How many frames to bridge (e.g., 5 frames = 50ms at 160 step)
    
        # 1. Initial Binary VAD
        frames = tf.signal.frame(y_true_wave, frame_length=hop_size, frame_step=hop_size)
        frame_rms = tf.sqrt(tf.reduce_mean(tf.square(frames), axis=-1) + 1e-8)
        is_speech_f = tf.cast(frame_rms > threshold, tf.float32) # [batch, frames]

        # 2. Fill Gaps (Morphological Dilation)
        # We expand the '1's into the '0's. If a '1' is anywhere in the window,
        # the whole window becomes '1'.
        mask_expanded = is_speech_f[..., tf.newaxis] 
        filled_mask = tf.nn.max_pool1d(
            mask_expanded, ksize=fill_window, strides=1, padding='SAME'
        )
        
        # Force back to strict 0 or 1 (though max_pool on 0/1 data is already 0/1)
        is_speech_f = tf.squeeze(filled_mask, axis=-1)

        # 3. Calculate Weighting for the '0' areas
        # Even if the mask is 0/1, we still need to weigh the '0' frames 
        # so they don't overpower the '1' frames in the loss function.
        L_s = tf.reduce_sum(is_speech_f, axis=-1, keepdims=True)
        total_frames = tf.cast(tf.shape(is_speech_f)[1], tf.float32)
        L_ns = total_frames - L_s
        
        # Calculate noise weight (W_ns)
        weight_ns = (L_s * target_ratio) / (L_ns + 1e-8)
        weight_ns = tf.minimum(weight_ns, 1.0) # Cap at 1.0

        # 4. Final Mask Assembly
        # frame_mask will be 1.0 where speech is, and weight_ns where it isn't.
        # The VAD decision itself (is_speech_f) remains strictly 0 or 1.
        frame_mask = is_speech_f * 1.0 + (1.0 - is_speech_f) * weight_ns

        # 5. Expand back to sample-level
        sample_mask = tf.repeat(frame_mask, repeats=hop_size, axis=-1)
        pad_len = tf.shape(y_true_wave)[1] - tf.shape(sample_mask)[1]
        sample_mask = tf.pad(sample_mask, [[0, 0], [0, pad_len]])
        return sample_mask

    def __call__(self, y_true, y_pred, clean=None):
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
        max_val  = tf.reduce_max(
                tf.abs(wav_true),
                axis=1,
                keepdims=True)

        scale = tf.where(
            max_val > 1e-3,
            1.0 / max_val,
            1.0)

        wav_true = wav_true * scale
        wav_pred = wav_pred * scale
        losses = []

        # sample_mask = self._compute_ratio_balanced_vad(wav_true, target_ratio=0)
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(12,6))
        # plt.subplot(2,1,1)
        # plt.title("True Waveform")
        # plt.plot(wav_true[0,:].numpy())
        # plt.subplot(2,1,2)
        # plt.title("sample_mask Waveform")
        # plt.plot(sample_mask[0,:].numpy())
        # plt.show()

        # Compute loss at each STFT resolution
        for cfg in self.stft_configs:

            spec_true = self.stft(wav_true, **cfg)
            spec_pred = self.stft(wav_pred, **cfg)
            hop_size = cfg['hop_length']
            if clean is not None:
                num_lookahead = 2
                vad = self._compute_binary_filled_vad(
                    hop_size,
                    clean,
                    threshold=1e-4,
                    target_ratio=0.025,
                    fill_window=5)
                vad_frame = tf.concat([
                    tf.zeros((vad.shape[0],num_lookahead*160), dtype=vad.dtype),
                    vad
                ], axis=1)[:, :: hop_size]
                vad_frame=vad_frame[:, :tf.shape(spec_true)[1]]
                vad_frame = tf.expand_dims(vad_frame, axis=-1)

                # import matplotlib.pyplot as plt
                
                # pspec = 10 * tf.math.log(tf.abs(spec_true[0,:,:])**2 + 1e-12)
                
                
                # plt.figure(figsize=(12,6))
                
                # plt.title("Spectrogram with VAD")
                # plt.imshow(
                #     pspec.numpy().T,
                #     origin='lower',
                #     aspect='auto',
                #     cmap='pink_r',
                #     vmin=-80,
                #     vmax=10)
                # plt.plot(vad_frame[0,:,0].numpy() * 100, linewidth=2)
                
    
                # plt.colorbar()
                # plt.show()
            phase_true = complex_angle(spec_true, self.eps)
            phase_pred = complex_angle(spec_pred, self.eps)

            mag_true = complex_magnitude(spec_true, self.eps)
            mag_pred = complex_magnitude(spec_pred, self.eps)

            steps = tf.shape(mag_pred)[0] * tf.shape(mag_pred)[1]

            comp_true = polar_to_complex(tf.pow(mag_true, self.exp), phase_true)
            comp_pred = polar_to_complex(tf.pow(mag_pred, self.exp), phase_pred)
            if 0:
                abs_err = tf.abs(comp_pred - comp_true)
                if clean is not None:
                    abs_err = abs_err * vad_frame
                else:
                    abs_err = abs_err
                loss = tf.reduce_sum(tf.square(abs_err)) / tf.cast(steps, tf.float32)
            else:
                # Apply 3x3 averaging filter to smooth the error
                err_square = tf.abs(comp_pred - comp_true)**2 + 1e-12 # (B, T, F)
                
                err_square = tf.expand_dims(err_square, axis=-1)  # (B,T,F, 1)

                

                err_group = self.group_filter(err_square)   # (B,T,F, 1)
                err_group = tf.sqrt(err_group) # (B, T, F,1)
                
                err_group = err_group[..., 0]  # (B,T,F)
                if clean is not None:
                    err_group = err_group * vad_frame
                else:
                    err_group = err_group
                loss = tf.reduce_sum(err_group) / tf.cast(steps, tf.float32)
            losses.append(loss)

        # Average loss across all resolutions
        return tf.reduce_mean(losses)
