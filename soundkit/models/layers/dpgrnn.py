import tensorflow as tf
from tensorflow.keras import layers, Model

class GRNN(layers.Layer):
    """Simple wrapper for GRU (can be bidirectional)."""
    def __init__(self, output_size, bidirectional=False, unroll=False, **kwargs):
        super(GRNN, self).__init__(**kwargs)
        self.bidirectional = bidirectional
        if bidirectional:
            self.rnn = layers.Bidirectional(
                layers.GRU(output_size, return_sequences=True, unroll=unroll)
            )
        else:
            self.rnn = layers.GRU(output_size, return_sequences=True, return_state=True, unroll=unroll)

    def call(self, x, states=None):
        # x: (B, T, C)
        if self.bidirectional:
            return self.rnn(x)  # (B, T, 2H)
        else:
            y, states = self.rnn(x, initial_state=states)  # (B, T, H), (B, H)
            return y, states

class DPGRNN(tf.keras.layers.Layer):
    """Grouped Dual-Path GRU (TensorFlow version)
       Input:  (B, T, F, C)
       Output: (B, T, F, C)
    """
    def __init__(self, num_chs, num_freqs, batchsize, unroll=False, **kwargs):
        super(DPGRNN, self).__init__(**kwargs)
        self.num_chs = num_chs
        self.num_freqs = num_freqs
        self.hidden_size = num_chs
        self.batchsize = batchsize

        # Intra GRU: across frequency (F)
        self.intra_rnn = GRNN(output_size=num_chs//2, bidirectional=True, unroll=unroll)
        self.intra_fc = layers.Dense(num_chs)
        # self.intra_ln = layers.LayerNormalization(epsilon=1e-8)

        # Inter GRU: across time (T)
        self.inter_rnn = GRNN(output_size=num_chs, bidirectional=False, unroll=unroll)
        self.inter_fc = layers.Dense(num_chs)

        self.states = tf.Variable(
            initial_value=tf.zeros([num_freqs * batchsize, num_chs]),
            trainable=False
        )
        self.reset_states()
        # self.inter_ln = layers.LayerNormalization(epsilon=1e-8)

    def call(self, x):
        """
        x: (B, T, F, C)
        """
        B, T, F, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        # ----- Intra GRU (across frequency) -----
        # (B, T, F, C) -> (B*T, F, C)
        
        intra_x = tf.reshape(x, [B * T, F, C])   # (B*T, F, C)
        intra_x = self.intra_rnn(intra_x)        # (B*T, F, H)
        intra_x = self.intra_fc(intra_x)         # (B*T, F, H)

        # intra_x = self.intra_ln(intra_x)
        intra_x = tf.reshape(intra_x, [B, T, F, self.hidden_size])  # (B, T, F, H)
        intra_out = x + intra_x  # (B,T,F,H)

        # ----- Inter GRU (across time) -----
        # (B,T,F,C) -> (B*F, T, C)
        inter_x = tf.transpose(intra_out, perm=[0, 2, 1, 3])   # (B, F, T, H)
        inter_x = tf.reshape(inter_x, [B * F, T, C])           # (B*F, T, C)
        inter_x, states = self.inter_rnn(inter_x, self.states) # (B*F, T, H)
        self.states.assign(states)
        inter_x = self.inter_fc(inter_x)                       # (B*F, T, H)
        inter_x = tf.reshape(
            inter_x, [B, F, T, self.hidden_size])              # (B, F, T, H)
        inter_x = tf.transpose(inter_x, perm=[0, 2, 1, 3])     # (B, T, F, H)
        # inter_x = self.inter_ln(inter_x)
        inter_out = intra_out + inter_x                        # (B,T,F,H)

        return inter_out  # (B,T,F,H)

    def reset_states(self, zero_state=False):
        
        if zero_state:
            self.states.assign(tf.zeros_like(self.states))
        else:
            self.states.assign(
                tf.random.uniform(
                    [self.num_freqs * self.batchsize, self.hidden_size]))