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
import numpy as np
from soundkit.utils.tf_complex_utils import (
    polar_to_complex,
    complex_angle,
    complex_magnitude,
    get_compressed_complex,
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
                 hop_size=None,
                 exp_features = 0.5,
                 num_lookahead=0,
                 eps=1e-8,
                 name="multi_resolution_stft_loss",
                 **kwargs):
        super().__init__(name=name)
        # Allow hop_size as alias for hop_length (from config)

        if hop_size is not None:
            hop_length = hop_size
        # Default STFT configs for multiple time-frequency resolutions
        if stft_configs is None:
            stft_configs = [
                {"n_fft": 256, "hop_length": 80, "win_length": 160},    # 5ms
                {"n_fft": 512, "hop_length": 160, "win_length": 320},   # 10ms
                {"n_fft": 1024, "hop_length": 320, "win_length": 640},  # 20ms
                {"n_fft": 2048, "hop_length": 640, "win_length": 1280}, # 40ms
            ]
        self.exp_features = exp_features
        self.num_lookahead = num_lookahead
        self.eps = eps  # Small value to avoid log/zero issues
        self.stft_configs = stft_configs
        self.fft_size = fft_size
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.group_shape = (3, 3)  # Shape for the averaging filter
        self.group_filter = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=self.group_shape,
            trainable=False,
            use_bias=False,
            padding='SAME',
            kernel_initializer=tf.keras.initializers.Constant(1.0 / (self.group_shape[0] * self.group_shape[1]))
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
        fill_window=5,): # How many frames to bridge (e.g., 5 frames = 50ms at 160 step)

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
        L_s = tf.minimum(L_ns, L_s)
        # Calculate noise weight (W_ns)
        # Only calculate weight if there is actually noise to weight
        # If L_ns is 0, weight_ns doesn't matter (multiply by zero later), 
        # but we set it to 1.0 to avoid 0/0.
        weight_ns = tf.where(
            L_ns > 0,
            (L_s * target_ratio) / (L_ns + 1e-8),
            1.0
        )
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

    def __call__(self, y_true, y_pred, clean=None, lengths=None, **kwargs):
        """
        Compute multi-resolution STFT loss between predicted and true complex STFTs.

        Args:
            y_true (tf.Tensor): Ground truth complex STFTs, shape (B, T, F).
            y_pred (tf.Tensor): Predicted complex STFTs, shape (B, T, F).

        Returns:
            tf.Tensor: Scalar loss averaged over all resolutions.
        """

        # Convert STFTs back to time-domain waveforms
        if self.exp_features != 1.0:
            y_true = get_compressed_complex(y_true, 1/self.exp_features, self.eps)
            y_pred = get_compressed_complex(y_pred, 1/self.exp_features, self.eps)
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

            mask_oa = tf.abs(spec_true) > tf.abs(spec_pred)
            mask_oa = tf.cast(mask_oa, tf.float32)

            hop_size = cfg['hop_length']
            if clean is not None:
                vad = self._compute_binary_filled_vad(
                    hop_size,
                    clean * scale,
                    threshold=1e-4,
                    target_ratio=1e-3, # 1e-3
                    fill_window=5)

                vad_frame = tf.concat([
                    tf.zeros((vad.shape[0],self.num_lookahead*self.hop_length), dtype=vad.dtype),
                    vad
                ], axis=1)[:, :: hop_size]

                vad_frame=vad_frame[:, :tf.shape(spec_true)[1]]
                vad_frame = tf.expand_dims(vad_frame, axis=-1)
                # mask_oa *= vad_frame
                if 0:
                    import matplotlib.pyplot as plt
                    tmp = tf.abs(get_compressed_complex(
                        spec_true,
                        self.exp_features,
                        self.eps))
                    
                    plt.figure(figsize=(12,6))
                    plt.subplot(4,1,1)
                    plt.title("Spectrogram with VAD (0.3 power)")
                    tf.print(self.exp_features)
                    tmp = tmp[0]
                    
                    plt.imshow(
                        tmp.numpy().T,
                        origin='lower',
                        aspect='auto',
                        cmap='pink_r',
                        vmin=tmp.numpy().min(),
                        vmax=tmp.numpy().max())
                    plt.plot(vad_frame[0,:,0].numpy() * 100, linewidth=2)
                    plt.colorbar()

                    plt.subplot(4,1,2)
                    plt.title(f"Spectrogram ({self.exp_features} power) with VAD`")
                    tmp = tf.abs(get_compressed_complex(
                        spec_true,
                        self.exp_features,
                        self.eps))
                    tmp = tmp[0]
                    
                    plt.imshow(
                        tmp.numpy().T,
                        origin='lower',
                        aspect='auto',
                        cmap='pink_r',
                        vmin=tmp.numpy().min(),
                        vmax=tmp.numpy().max())
                    plt.plot(vad_frame[0,:,0].numpy() * 100, linewidth=2)
                    plt.colorbar()

                    plt.subplot(4,1,3)
                    plt.imshow(
                        mask_oa[0,:,:].numpy().T,
                        origin='lower',
                        aspect='auto',
                        cmap='pink_r',
                        vmin=0,
                        vmax=1)
                    plt.colorbar()

                    plt.subplot(4,1,4)
                    plt.plot(wav_true[0,:].numpy(), label='True Waveform')
                    plt.ylim([-1, 1])
                    plt.show()
            # phase_true = complex_angle(spec_true, self.eps)
            # phase_pred = complex_angle(spec_pred, self.eps)
            # phase_true = tf.math.angle(spec_true)
            # phase_pred = tf.math.angle(spec_pred)
            
            # mag_true = complex_magnitude(spec_true, self.eps)
            # mag_pred = complex_magnitude(spec_pred, self.eps)

            steps = tf.shape(spec_true)[0] * tf.shape(spec_true)[1] * tf.shape(spec_true)[2]



            if self.exp_features == 1.0:
                comp_true = spec_true
                comp_pred = spec_pred
            else:
                comp_true = get_compressed_complex(spec_true, self.exp_features, self.eps)
                comp_pred = get_compressed_complex(spec_pred, self.exp_features, self.eps)
            comp_mag_true = tf.abs(comp_true)
            comp_mag_pred = tf.abs(comp_pred)


            def smooth_to_l1(
                    err_sq: tf.Tensor,
                    filter_layer: tf.keras.layers.Layer,
                    upper_bound: int = None,
                    eps: float = 1e-7):
                """ 
                Convert squared error to a smooth L1-like error with low-frequency smoothing.
                """
                # Split 
                if upper_bound is not None:
                    low = err_sq[:, :, :upper_bound]
                    high = err_sq[:, :, upper_bound:]

                    # Smooth the low frequencies in the squared domain
                    low_smoothed_sq = filter_layer(tf.expand_dims(low, axis=-1))[..., 0]

                    # Recombine
                    total_sq = tf.concat([low_smoothed_sq, high], axis=2)
                else:
                    # Smooth the entire spectrum
                    total_sq = filter_layer(tf.expand_dims(err_sq, axis=-1))[..., 0]
                # CONVERT TO L1: This is the "Soft-L1" stabilization
                # sqrt(x^2 + eps) is smooth at zero and looks like L1 elsewhere
                return tf.sqrt(total_sq + eps)
            
            def asymmetric_weight_mask(y_true, y_pred, alpha=2.0):
                # 1. Condition: is target magnitude > predicted magnitude? (Over-attenuation)
                # Result is a boolean tensor: True where we are "too quiet"
                is_oa = tf.abs(y_true) > tf.abs(y_pred)

                # 2. Convert boolean to float (1.0 for OA, 0.0 for not OA)
                mask_float = tf.cast(is_oa, tf.float32)

                # 3. Scale to tunable weights
                # If is_oa is 1.0 -> (1.0 * (alpha - 1)) + 1 = alpha
                # If is_oa is 0.0 -> (0.0 * (alpha - 1)) + 1 = 1.0
                weights = mask_float * (alpha - 1.0) + 1.0

                return weights
    
            def smooth_asymmetric_weight_mask(y_true, y_pred, alpha=3.0, k=1.0, eps=1e-7):
                """
                Smoothly penalizes over-attenuation using a scaled Sigmoid.
                k controls the steepness of the transition.
                """
                # 1. Calculate the difference
                mag_true = tf.abs(y_true)
                mag_pred = tf.abs(y_pred)
                
                # 1. Calculate absolute difference
                diff = mag_true - mag_pred
                diff = ((diff - 0.01) / (mag_true + eps))
                # 2. Zero out negative differences (under-attenuation / noise)
                relu_diff = tf.nn.relu(diff)

                # 3. Apply the slope (k) and cap the maximum penalty
                                # The maximum penalty we can add is (alpha - 1.0)
                max_penalty = alpha - 1.0
                penalty = tf.minimum(relu_diff * k, max_penalty)
                # Scale from [0, 1] to [1.0, alpha]
                # 4. Add to the baseline weight
                weights = 1.0 + penalty

                return weights

            mask_oa = smooth_asymmetric_weight_mask(
                    tf.abs(spec_true),
                    tf.abs(spec_pred),
                    alpha=1.2, # 2
                    k=4.0)

            # Scaling
            norm = tf.reduce_max(tf.abs(wav_true), axis=-1, keepdims=True)
            norm_r = tf.minimum(1.0 / (norm + 1e-4), 100.0)
            norm_r = tf.expand_dims(norm_r, axis=-1)
            if 0:
                err = tf.abs(comp_pred - comp_true)**2
                err_mag = tf.abs(comp_mag_pred - comp_mag_true)**2
                # if clean is not None:
                if 1:
                    err *= vad_frame
                    err_mag *= vad_frame
                loss2 = tf.reduce_sum(tf.sqrt(err + self.eps) * mask_oa)
                loss4 = tf.reduce_sum(tf.sqrt(err_mag + self.eps) * mask_oa)
                loss = (loss2  + loss4) / 2.0 / tf.cast(steps, tf.float32)
            else:
                # 1. Complex Difference Square
                # We start with squared distance because it's differentiable at the origin
                err_sq_cmplx = tf.abs(comp_pred - comp_true)**2

                # 2. Magnitude Difference Square
                err_sq_mag = (tf.abs(comp_pred) - tf.abs(comp_true))**2

                # Split and Smooth (using your low-pass logic)
                upper_bound = int((5000 / 16000) * cfg['n_fft'])

                # Apply the stable L1 conversion
                err_l1_cmplx = smooth_to_l1(
                    err_sq_cmplx,
                    filter_layer = self.group_filter,
                    # upper_bound = upper_bound,
                    eps = self.eps)

                err_l1_mag   = smooth_to_l1(
                    err_sq_mag,
                    filter_layer = self.group_filter,
                    # upper_bound = upper_bound,
                    eps = self.eps)

                # Apply VAD and Norm
                if 1: # Your existing logic
                    # if clean is not None:
                    #     err_l1_cmplx *= vad_frame
                    #     err_l1_mag   *= vad_frame
                    if lengths is not None:
                        lookahead_samples = self.num_lookahead * self.hop_length
                        frame_lengths = (lengths - lookahead_samples) // cfg['hop_length']
                        frame_lengths = tf.maximum(frame_lengths, 0)
                        masks = tf.sequence_mask(
                            frame_lengths,
                            maxlen=tf.shape(err_l1_cmplx)[1])
                        err_l1_cmplx = tf.boolean_mask(err_l1_cmplx, masks)
                        err_l1_mag   = tf.boolean_mask(err_l1_mag, masks)
                        mask_oa = tf.boolean_mask(mask_oa, masks)

                # Final Loss Summation (L1 Style)
                # mask_oa *=3.0 # Your existing scaling for the OA mask

                # loss1 = tf.reduce_sum(err_l1_cmplx)
                loss2 = tf.reduce_sum(err_l1_cmplx * mask_oa)
                # loss3 = tf.reduce_sum(err_l1_mag)
                loss4 = tf.reduce_sum(err_l1_mag * mask_oa)

                valid_time_freq_steps = tf.maximum(tf.size(err_l1_cmplx), 1)
                loss = (loss2 + loss4) / 2.0 / tf.cast(valid_time_freq_steps, tf.float32)
            losses.append(loss)

        # Average loss across all resolutions
        loss = tf.reduce_mean(losses)
        return tf.debugging.check_numerics(loss, message="Found it!")
