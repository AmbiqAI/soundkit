import numpy as np
import tensorflow as tf

class ERB(tf.keras.layers.Layer):
    """ Equivalent Rectangular Bandwidth (ERB) filterbank layer
    Args:
        erb_subband_1: Number of unchanged low-frequency subbands
        erb_subband_2: Number of ERB-mapped high-frequency subbands
        nfft: FFT size
        high_lim: High frequency limit for ERB mapping (Hz)
        fs: Sampling frequency (Hz)
    """
    def __init__(
            self,
            erb_subband_1 = 65,
            erb_subband_2 = 64,
            nfft=512,
            high_lim=8000,
            fs=16000,
            platform: str = "tensorflow",):
        super().__init__()
        self.erb_subband_1 = erb_subband_1
        self.erb_subband_2 = erb_subband_2
        self.platform = platform
        # Create filterbank: shape (erb_subband_2, F_high)
        erb_filters = self.erb_filter_banks(
            erb_subband_1, erb_subband_2, nfft, high_lim, fs
        ).astype(np.float32)
        
        nfreqs = nfft // 2 + 1
        # f_high = nfreqs - erb_subband_1

        # # Fixed (non-trainable) filterbank
        # self.erb_fc = tf.keras.layers.Dense(
        #     erb_subband_2,
        #     use_bias=False,
        #     trainable=False,
        # )
        # self.ierb_fc = tf.keras.layers.Dense(
        #     f_high,
        #     use_bias=False,
        #     trainable=False,
        # )

        # # Set weights (Dense kernel is shape (in, out))
        # self.erb_fc.build((None, f_high))
        # self.ierb_fc.build((None, erb_subband_2))
        
        # erb_filters_norm = erb_filters / tf.sqrt(
        #     tf.reduce_sum(erb_filters**2, axis=1, keepdims=True))

        # self.erb_fc.kernel.assign(
        #     tf.transpose(erb_filters_norm, perm=[1, 0]))       # PyTorch had (out, in)

        # erb_filters_norm = erb_filters / tf.sqrt(
        #     tf.reduce_sum(erb_filters**2, axis=0, keepdims=True))
        # self.ierb_fc.kernel.assign(erb_filters_norm)        # transpose

        lst = []

        for i in range(erb_subband_1):
            tmp = np.zeros((nfreqs,))
            tmp[i] = 1.0
            lst.append(tmp)
        for i in range(erb_subband_2):
            tmp = np.zeros((nfreqs,))
            tmp[erb_subband_1:] = erb_filters[i]
            lst.append(tmp)
        tmp = np.stack(lst, axis=-1)
        mat_whole = tf.constant(tmp, dtype=tf.float32)

        self.filter_map = mat_whole / tf.sqrt(
            tf.reduce_sum(mat_whole**2, axis=0, keepdims=True))

        self.filter_inv_map = tf.transpose(mat_whole, perm=[1,0])
        self.filter_inv_map = self.filter_inv_map / tf.sqrt(
            tf.reduce_sum(self.filter_inv_map**2, axis=0, keepdims=True))

        if self.platform == "numpy":
            self.filter_map = self.filter_map.numpy()
            self.filter_inv_map = self.filter_inv_map.numpy()

    # ---------------------------
    # ERB <-> Hz conversion
    # ---------------------------
    def hz2erb(self, freq):
        return 21.4 * np.log10(1 + 0.00437 * freq)

    def erb2hz(self, erb_f):
        return (10 ** (erb_f / 21.4) - 1) / 0.00437

    # ---------------------------
    # Filterbank construction
    # ---------------------------
    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft, high_lim, fs):
        low_lim = erb_subband_1 / nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)

        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points) / fs * nfft).astype(np.int32)

        num_bins = nfft // 2 + 1
        filters = np.zeros((erb_subband_2, num_bins), dtype=np.float32)

        # First band
        filters[0, bins[0]:bins[1]] = \
            (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) / \
            (bins[1] - bins[0] + 1e-12)

        # Middle bands
        for i in range(erb_subband_2 - 2):
            # Rising slope
            filters[i+1, bins[i]:bins[i+1]] = \
                (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12) / \
                (bins[i+1] - bins[i] + 1e-12)
            # Falling slope
            filters[i+1, bins[i+1]:bins[i+2]] = \
                (bins[i+2] - np.arange(bins[i+1], bins[i+2]) + 1e-12) / \
                (bins[i+2] - bins[i+1] + 1e-12)

        # Last band
        filters[-1, bins[-2]:bins[-1]+1] = \
            1 - filters[-2, bins[-2]:bins[-1]+1]

        # Remove low frequencies (unchanged band)
        filters = filters[:, erb_subband_1:]
        return filters

    # ---------------------------
    # BAND MAPPING: STFT → ERB
    # x: (B,C,T,F)
    # ---------------------------
    def bm(self, x):
        """ Band mapping from STFT to ERB
        Args:
            x: Tensor of shape (B, C, T, F)
        Returns:
            Tensor of shape (B, C, T, F_erb)
        """
        # x_low = x[..., :self.erb_subband_1]
        # x_high = x[..., self.erb_subband_1:]  # shape (..., F_high)
        # # Dense works on last dimension only
        # x_erb = self.erb_fc(x_high)
        # return tf.concat([x_low, x_erb], axis=-1)

        if self.platform == "numpy":
            return np.matmul(
                x,
                self.filter_map,
            )
        else:
            return tf.matmul(
                x,
                self.filter_map,
            )
    # ---------------------------
    # BAND SYNTHESIS: ERB → STFT
    # x_erb: (B,C,T,F_erb)
    # ---------------------------
    def bs(self, x_erb):
        """ Band synthesis from ERB to STFT
        Args:
            x_erb: Tensor of shape (B, C, T, F_erb)
        Returns:
            Tensor of shape (B, C, T, F)
        """
        # x_low = x_erb[..., :self.erb_subband_1]
        # x_erb_high = x_erb[..., self.erb_subband_1:]

        # x_stft_high = self.ierb_fc(x_erb_high)

        # return tf.concat([x_low, x_stft_high], axis=-1)
        if self.platform == "numpy":
            return np.matmul(
                x_erb,
                self.filter_inv_map,
            )
        else:
            return tf.matmul(
                x_erb,
                self.filter_inv_map,
            )
if __name__ == "__main__":
    erb = ERB(erb_subband_1=65, erb_subband_2=64)

    
    x = tf.random.normal([4, 2, 100, 257])  # (B,C,T,F)

    x_erb = erb.bm(x)
    x_rec = erb.bs(x_erb)
