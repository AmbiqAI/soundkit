
import tensorflow as tf
from . import UNetParams, SliceLayer, get_unet_info
from ..layers.TransposeConv2D import TransposeConv2D, SeparableTransposeConv2D
class decoder_unet(tf.keras.layers.Layer):
    """ Decoder of UNet"""
    def __init__(
            self,
            params: UNetParams = UNetParams(),
            **kwargs):
        super(decoder_unet,self).__init__(**kwargs)
        self.params = params
        self.freq_bins, self.pad_freq_bins =  get_unet_info(
            params.num_chs,
            dim_feat=params.dim_feat,
            kernel_size_freq=params.kernel_size_freq)
        # self.freq_bins = [257, 128, 63, 31, 15]
        self.skip_connection = params.skip_connection
        self.convs=[]
        stages = len(params.num_chs) - 1
        self.states=self.make_states()
        # input shape (batch, T, Freq, 1)
        # self.pad_freq_bins = [0,1, 0, 0]
        self.zeros = []
        if params.dropout > 0:
            self.skip_dropout = tf.keras.layers.Dropout(
                rate=params.dropout,
                name='skip_dropout')
        else:
            self.skip_dropout = None

        for i, num_ch, num_ch_in in zip(range(stages), params.num_chs[:-1], params.num_chs[1:]):
            num_pad = self.pad_freq_bins[i]
            layer=tf.keras.Sequential(name=f"decoder_{i}")
            if i == 0:
               activation_layer=params.activation_final
            else:
                activation_layer = params.activation
            skip_ch_mult = 2 if params.skip_connection == 'concat' else 1

            if params.separable:

                layer.add(
                    SeparableTransposeConv2D(
                        filters=num_ch,
                        kernel_size=(params.kernel_size_time, params.kernel_size_freq),
                        activation=activation_layer,
                        num_channels_in=num_ch_in * skip_ch_mult,
                        batch_size=params.batchsize,
                        time_steps=params.time_steps,
                        normalization_layer=params.normalization_layer,
                        depthwise_activation=params.depthwise_activation,
                        name=f"conv_tran_{i}"
                        ))
            else:
                if 0:
                    layer.add(
                        TransposeConv2D(
                            filters=num_ch,
                            kernel_size=(params.kernel_size_time, params.kernel_size_freq),
                            activation=params.activation,
                            depthwise_activation=params.depthwise_activation,
                            name=f"conv_tran_{i}"
                            ))
                else:

                    layer.add(
                        tf.keras.layers.Conv2DTranspose(
                            filters=num_ch,
                            kernel_size=(params.kernel_size_time, params.kernel_size_freq),
                            strides=(1, 2),
                            padding='valid',
                            activation=activation_layer,
                            kernel_initializer='he_normal',
                            name=f"conv_tran_{i}"
                            ))
                    layer.add(
                        SliceLayer(
                            params.kernel_size_time,
                            name=f"slice_{i}"))
            if num_pad > 0:
                zeros=tf.zeros((params.batchsize, params.time_steps, num_pad, num_ch), dtype=tf.float32)
                zeros = tf.Variable(zeros, trainable=False)
            else:
                zeros = None
  
            # layer.add(
            #     tf.keras.layers.ZeroPadding2D(
            #         padding=((0, 0),(0,num_pad)),
            #         name=f"padding_{i}")
            # )
            self.zeros = [zeros] + self.zeros # reverse the order
            self.convs = [layer] + self.convs # reverse the order

    def call(
            self,
            x,
            inputs_dec,
            # states=None,
            training=False):
        """ Forward pass"""

        for i, layer_info in enumerate(zip(inputs_dec[::-1], self.convs, self.states, self.zeros)):

            encode, net, state, zeros = layer_info
            state_en, state_de = state

            if self.params.kernel_size_time > 1:
                encode = tf.concat([state_en, encode], axis=1) # time concatenation
                x = tf.concat([state_de, x], axis=1) # time concatenation

                state_en_update=tf.identity(
                    encode[:,-(self.params.kernel_size_time-1):,:,:])
                state_de_update=tf.identity(
                    x[:,-(self.params.kernel_size_time-1):,:,:])

            if self.skip_dropout is not None:
                encode = self.skip_dropout(encode, training=training)
            if self.skip_connection == 'add':
                comb = encode + x
            else:
                comb = tf.concat([encode, x], axis=-1) # skip connection (channel concatenation)
            x = net(comb, training=training)

            # compensate the downsampling
            x = tf.concat([x, zeros], axis=2) if zeros is not None else x
            if self.params.kernel_size_time > 1:
                self.states[i][0].assign(state_en_update)
                self.states[i][1].assign(state_de_update)
        return x

    def make_states(self):
        """ Make states"""
        len_pad = self.params.kernel_size_time - 1

        states=[]
        for num_ch, freq_bin in zip(self.params.num_chs[1:], self.freq_bins[1:]):
            # reverse the order

            state_en = tf.zeros(
                (self.params.batchsize, len_pad, freq_bin, num_ch))

            state_en = tf.Variable(state_en, trainable=False) # for eager mode

            state_de = tf.zeros(
                (self.params.batchsize, len_pad, freq_bin, num_ch))

            state_de = tf.Variable(state_de, trainable=False) # for eager mode

            states = [(state_en, state_de)] + states

        return states

    def reset_states(self):
        """ Reset states"""
        for state in self.states:
            state[0].assign(state[0] * 0)
            state[1].assign(state[1] * 0)

    def shuffle(self, x1, x2):
        """Channel shuffle. x1, x2: (B,T,F,C). Output: (B,T,F,2C) interleaved."""
        x = tf.stack([x1, x2], axis=-1)   # (B,T,F,C,2)
        shape = tf.shape(x)
        return tf.reshape(x, [shape[0], shape[1], shape[2], -1])
