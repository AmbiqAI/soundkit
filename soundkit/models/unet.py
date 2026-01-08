"""
UNet model in the streaming mode
see https://bpb-us-w2.wpmucdn.com/u.osu.edu/dist/7/125945/files/2024/04/Tan-Wang.taslp20-b86daf0e06ac52f1.pdf
"""
import tensorflow as tf
from .unet_sublayers import UNetParams, get_unet_info
from .unet_sublayers.encoder import encoder_unet
from .unet_sublayers.decoder import decoder_unet
from .layers.dpgrnn import DPGRNN
from .layers.tcn import tcn
from soundkit.utils.converter_fix_point import fakefix_tf

class unet(tf.keras.Model):
    """ UNet"""
    def __init__(
            self,
            params: UNetParams = UNetParams(),
            **kwargs):
        super(unet,self).__init__(**kwargs)
        self.params = params
        self.output_scaling = params.output_scaling
        self.encoder = encoder_unet(
            params=params,)
        self.decoder = decoder_unet(
            params=params)

        self.freq_bins, self.pad_freq_bins = get_unet_info(
            params.num_chs,
            dim_feat=params.dim_feat)
        
        # import pdb; pdb.set_trace()
        # params.dim_out = params.dim_feat
        
        self.F=self.freq_bins[-1]
        self.chs=params.num_chs[-1]
        self.states = self.make_states()

        if params.bottleneck == 'lstm':
            self.rnn = tf.keras.layers.LSTM(
                self.F * self.chs,
                return_state=True,
                stateful=False,
                unroll=params.unroll_rnn,
                return_sequences=True)
        elif params.bottleneck == 'gru':
            self.rnn = tf.keras.layers.GRU(
                self.F * self.chs,
                return_state=True,
                stateful=False,
                unroll=params.unroll_rnn,
                return_sequences=True)
        elif params.bottleneck == 'dpgrnn':
            self.rnn = DPGRNN(
                num_chs=self.chs,
                num_freqs=self.F)
        elif params.bottleneck == "tcn":
            self.rnn = tcn(
                filters         = self.F * self.chs,
                kernel_size     = params.kernel_size_tcn,
                batchsize       = params.batchsize,
                dim_feat        = self.F * self.chs,
                dilations       = params.dilations,
                # dropout_rate    = dropout,
                second_dropout  = True,
                activation=params.activation  ,
                name            = "tcn_layer"
            )

        if not params.bypass_last_fc:
            if params.num_chs[0] == 1: # real case

                self.fc_real = tf.keras.layers.Dense(
                    params.dim_out,
                    activation='sigmoid',
                    name='fc_real')

                self.complex=False
            else: # complex case
                self.fc_real = tf.keras.layers.Dense(
                    params.dim_out,
                    activation='tanh',
                    name='fc_real')
                self.fc_imag = tf.keras.layers.Dense(
                    params.dim_out,
                    activation='tanh',
                    name='fc_imag')
                self.complex=True
        else:
            if params.num_chs[0] == 1:
                self.complex=False
            else:
                self.complex=True

        if params.dropout > 0:
            self.dropout = tf.keras.layers.Dropout(
                rate=params.dropout,
                name='dropout')

    def reset_states(self, zero_state=False):
        """ Reset states"""
        if self.params.bottleneck == "lstm":
            h_states = tf.Variable(
                                tf.random.uniform(
                                    [self.params.batchsize, self.F * self.chs],
                                    minval=-1,
                                    maxval=1 - 2**-8), # assume 8 bit quantization
                                dtype = tf.float32,
                                trainable = False)

            c_states = tf.Variable(
                                tf.random.uniform(
                                    [self.params.batchsize, self.F * self.chs],
                                    minval=-128,
                                    maxval=128 - 2**-8), # assume 8 bit quantization
                                dtype = tf.float32,
                                trainable = False)
            if zero_state:
                self.states[0].assign(h_states * 0)
                self.states[1].assign(c_states * 0)
            else:
                self.states[0].assign(h_states)
                self.states[1].assign(c_states)

        elif self.params.bottleneck == "gru":
            h_states = tf.Variable(
                                tf.random.uniform(
                                    [self.params.batchsize, self.F * self.chs],
                                    minval=-1,
                                    maxval=1),
                                dtype = tf.float32,
                                trainable = False)
            if zero_state:
                self.states[0].assign(h_states * 0)
            else:
                self.states[0].assign(h_states)
            
        elif self.params.bottleneck == "tcn":
            self.rnn.reset_states()
        self.encoder.reset_states()
        self.decoder.reset_states()

    def make_states(self, zero_state=False):
        """ Make states"""
        if self.params.bottleneck == "lstm":
            h_states = tf.Variable(
                                tf.random.uniform(
                                    [self.params.batchsize, self.F * self.chs],
                                    minval=-1,
                                    maxval=1 - 2**-8), # assume 8 bit quantization
                                dtype = tf.float32,
                                trainable = False)

            c_states = tf.Variable(
                                    tf.random.uniform(
                                        [self.params.batchsize, self.F * self.chs],
                                        minval=-128,
                                        maxval=128 - 2**-8), # assume 8 bit quantization
                                    dtype = tf.float32,
                                    trainable = False)
            if zero_state:
                h_states.assign(h_states * 0)
                c_states.assign(c_states * 0)

        elif self.params.bottleneck == "gru":
            h_states = tf.Variable(
                                tf.random.uniform(
                                    [self.params.batchsize, self.F * self.chs],
                                    minval=-1,
                                    maxval=1),
                                dtype = tf.float32,
                                trainable = False)
            if zero_state:
                h_states.assign(h_states * 0)
            c_states = None
        return h_states, c_states

    def call(
            self,
            inputs,
            training=False):
        """ Forward pass"""
        if not self.complex:
            inputs = tf.expand_dims(inputs, axis=-1)

        x = inputs

        # states_de = states

        # encoder
        outputs = self.encoder(x)

        # bottleneck rnn

        if self.params.bottleneck == 'lstm':
            timesteps = tf.shape(outputs[-1])[1]
            out = tf.reshape(
                outputs[-1],
                (self.params.batchsize, timesteps, -1))
            if self.params.dropout > 0:
                out = self.dropout(out, training=training)

            out, h_state, c_state = self.rnn(out, initial_state=self.states)
            # h_state = tf.clip_by_value(h_state, -1, 1)  # assume 8 bit quantization
            # c_state = tf.clip_by_value(c_state, -128, 128 - 2**-8)  # assume 8 bit quantization
            self.states[0].assign(h_state)
            self.states[1].assign(c_state)
            input_dec = tf.reshape(
                out,
                (self.params.batchsize, timesteps, self.F, self.chs))
        elif self.params.bottleneck == 'gru':
            timesteps = tf.shape(outputs[-1])[1]
            out = tf.reshape(
                outputs[-1],
                (self.params.batchsize, timesteps, -1))
            if self.params.dropout > 0:
                out = self.dropout(out, training=training)

            out, h_state = self.rnn(out, initial_state=self.states[0])
            # h_state = tf.clip_by_value(h_state, -1, 1 - 2**-8)  # assume 8 bit quantization
            self.states[0].assign(h_state)
            input_dec = tf.reshape(
                out,
                (self.params.batchsize, timesteps, self.F, self.chs))
        elif self.params.bottleneck == 'dpgrnn':

            out = outputs[-1]  # (B, F, T, C)
            if self.params.dropout > 0:
                out = self.dropout(out, training=training)
            out = self.rnn(out)  # (B, F, T, C)
            input_dec = out
        elif self.params.bottleneck == 'tcn':
            timesteps = tf.shape(outputs[-1])[1]
            out = tf.reshape(
                outputs[-1],
                (self.params.batchsize, timesteps, -1))
            out = self.rnn(out)
            input_dec = tf.reshape(
                out,
                (self.params.batchsize, timesteps, self.F, self.chs))
        else:
            raise ValueError(
                f"Unsupported bottleneck type: {self.params.bottleneck}")
        # decoder
        output = self.decoder(
            input_dec,
            outputs)

        # final projection
        if not self.params.bypass_last_fc:
            if self.complex:
                output_real = self.fc_real(output[...,0])
                output_imag = self.fc_imag(output[...,1])
                output = tf.stack(
                    [output_real, output_imag],
                    axis=-1)
            else:
                output = self.fc_real(output[...,0])

        return output * self.output_scaling # scale output
