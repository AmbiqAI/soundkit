"""
UNet model in the streaming mode
see https://bpb-us-w2.wpmucdn.com/u.osu.edu/dist/7/125945/files/2024/04/Tan-Wang.taslp20-b86daf0e06ac52f1.pdf
"""
import tensorflow as tf
from copy import deepcopy
from .unet_sublayers import UNetParams, get_unet_info
from .unet_sublayers.encoder import encoder_unet
from .unet_sublayers.decoder import decoder_unet

class cunet(tf.keras.Model):
    """ UNet"""
    def __init__(
            self,
            params: UNetParams = UNetParams(),
            **kwargs):
        super(cunet,self).__init__(**kwargs)
        self.complex = True
        self.params = params

        self.encoder = encoder_unet(
            params=params,)
        
        params_decoder = deepcopy(params)
        params_decoder.num_chs[0] = 1  # Ensure the first channel is 1 for decoder
        self.decoder_real = decoder_unet(
            params=params_decoder)
        self.decoder_imag = decoder_unet(
            params=params_decoder)

        self.freq_bins, self.pad_freq_bins = get_unet_info(
            params.num_chs,
            dim_feat=params.dim_feat)
        params.dim_out = params.dim_feat

        self.F=self.freq_bins[-1]
        self.chs=params.num_chs[-1]
        self.states = self.make_states()
        self.rnn = tf.keras.layers.LSTM(
            self.F * self.chs,
            return_state=True,
            stateful=False,
            unroll=params.unroll_rnn,
            return_sequences=True)

        self.fc_real = tf.keras.layers.Dense(
            params.dim_out,
            activation='tanh',
            name='fc_real')
        self.fc_imag = tf.keras.layers.Dense(
            params.dim_out,
            activation='tanh',
            name='fc_imag')

    def reset_states(self, zero_state=False):
        """ Reset states"""
        h_states = tf.Variable(
                            tf.random.uniform(
                                [self.params.batchsize, self.F * self.chs],
                                minval=-1,
                                maxval=1),
                            dtype = tf.float32,
                            trainable = False)

        c_states = tf.Variable(
                            tf.random.truncated_normal(
                                [self.params.batchsize, self.F * self.chs],
                                mean=0.0,
                                stddev=tf.sqrt(1 / self.F * self.chs)),
                            dtype = tf.float32,
                            trainable = False)
        if zero_state:
            self.states[0].assign(h_states * 0)
            self.states[1].assign(c_states * 0)
        else:
            self.states[0].assign(h_states)
            self.states[1].assign(c_states)

        self.encoder.reset_states()
        self.decoder.reset_states()

    def make_states(self, zero_state=False):
        """ Make states"""

        h_states = tf.Variable(
                            tf.random.uniform(
                                [self.params.batchsize, self.F * self.chs],
                                minval=-1,
                                maxval=1),
                            dtype = tf.float32,
                            trainable = False)

        c_states = tf.Variable(
                            tf.random.truncated_normal(
                                [self.params.batchsize, self.F * self.chs],
                                mean=0.0,
                                stddev=tf.sqrt(1 / self.F * self.chs)),
                            dtype = tf.float32,
                            trainable = False)
        if zero_state:
            h_states.assign(h_states * 0)
            c_states.assign(c_states * 0)

        return h_states, c_states

    def call(
            self,
            inputs,
            reset_input=tf.constant([0.0], dtype=tf.float32),
            # states=None,
            training=False):
        """ Forward pass"""

        x = inputs

        # states_de = states

        # encoder
        outputs = self.encoder(x)

        # bottleneck rnn
        timesteps = tf.shape(outputs[-1])[1]

        out = tf.reshape(
            outputs[-1],
            (self.params.batchsize, timesteps, -1))
        if self.params.dropout > 0:
            out = self.dropout(out, training=training)

        out, h_state, c_state = self.rnn(out, initial_state=self.states)

        self.states[0].assign(h_state)
        self.states[1].assign(c_state)
        input_dec = tf.reshape(
            out,
            (self.params.batchsize, timesteps, self.F, self.chs))

        # decoder
        output_real = self.decoder_real(
            input_dec,
            outputs)

        output_imag = self.decoder_imag(
            input_dec,
            outputs)

        # final projection
        output_real = self.fc_real(output_real[...,0])
        output_imag = self.fc_imag(output_imag[...,0])
        output = tf.stack([output_real, output_imag], axis=-1)
        return output
