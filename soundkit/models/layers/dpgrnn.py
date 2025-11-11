import tensorflow as tf
from tensorflow.keras import layers, Model

class GRNN(layers.Layer):
    """Simple wrapper for LSTM (can be bidirectional)."""
    def __init__(self, output_size, bidirectional=False, **kwargs):
        super(GRNN, self).__init__(**kwargs)
        self.bidirectional = bidirectional
        if bidirectional:
            self.rnn = layers.Bidirectional(
                layers.LSTM(output_size, return_sequences=True)
            )
        else:
            self.rnn = layers.LSTM(output_size, return_sequences=True)

    def call(self, x):
        # x: (B, T, C)
        return self.rnn(x)  # (B, T, H) or (B, T, 2H) if bidirectional


class DPGRNN(tf.keras.layers.Layer):
    """Grouped Dual-Path LSTM (TensorFlow version)
       Input:  (B, F, T, C)
       Output: (B, F, T, C)
    """
    def __init__(self, num_chs, num_freqs, **kwargs):
        super(DPGRNN, self).__init__(**kwargs)
        self.num_chs = num_chs
        self.num_freqs = num_freqs
        self.hidden_size = num_chs

        # Intra LSTM: across frequency (F)
        self.intra_rnn = GRNN(output_size=num_chs//2, bidirectional=True)
        self.intra_fc = layers.Dense(num_chs)
        self.intra_ln = layers.LayerNormalization(epsilon=1e-8)

        # Inter LSTM: across time (T)
        self.inter_rnn = GRNN(output_size=num_chs, bidirectional=False)
        self.inter_fc = layers.Dense(num_chs)
        self.inter_ln = layers.LayerNormalization(epsilon=1e-8)

    def call(self, x):
        """
        x: (B, F, T, C)
        """
        B, F, T, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        # ----- Intra LSTM (across frequency) -----
        # (B, F, T, C) -> (B*T, F, C)
        intra_x = tf.transpose(x, perm=[0, 2, 1, 3])       # (B, T, F, C)
        intra_x = tf.reshape(intra_x, [B * T, F, C])       # (B*T, F, C)
        intra_x = self.intra_rnn(intra_x)                  # (B*T, F, H)
        intra_x = self.intra_fc(intra_x)                   # (B*T, F, H)
        intra_x = tf.reshape(intra_x, [B, T, F, self.hidden_size])
        intra_x = self.intra_ln(intra_x)
        intra_out = x + tf.transpose(intra_x, perm=[0, 2, 1, 3])  # (B,F,T,H)

        # ----- Inter LSTM (across time) -----
        # (B,F,T,C) -> (B*F, T, C)
        inter_x = tf.reshape(intra_out, [B * F, T, C])
        inter_x = self.inter_rnn(inter_x)                  # (B*F, T, H)
        inter_x = self.inter_fc(inter_x)
        inter_x = tf.reshape(inter_x, [B, F, T, self.hidden_size])
        inter_x = self.inter_ln(inter_x)
        inter_out = intra_out + inter_x                    # residual

        return inter_out  # (B,F,T,H)
