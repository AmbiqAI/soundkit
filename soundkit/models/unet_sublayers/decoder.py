
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
            kernel_size_freq=params.kernel_size_freq,
            dim_feat=params.dim_feat)
        # self.freq_bins = [257, 128, 63, 31, 15]
        self.convs=[]
        stages = len(params.num_chs) - 1
        self.states=self.make_states()
        # input shape (batch, T, Freq, 1)
        # self.pad_freq_bins = [0,1, 0, 0]
        self.zeros = []
        self.dropout_layers = []
        self.dropout_skip_layers = []
        for i, num_ch, num_ch_in in zip(range(stages), params.num_chs[:-1], params.num_chs[1:]):
            num_pad = self.pad_freq_bins[i]
            layer=tf.keras.Sequential(name=f"decoder_{i}")
            if i == 0:
               activation_layer=params.output_activation
            else:
                activation_layer = params.activation

            # dropout settings
            # if params.dropout > 0:
            if 1:
                if i == stages - 1:
                    self.dropout_layers = [
                        tf.keras.layers.Dropout(params.dropout)
                        ] + self.dropout_layers
                else:
                    self.dropout_layers = [
                        tf.keras.layers.SpatialDropout2D(params.dropout)
                        ] + self.dropout_layers
            else:
                self.dropout_layers = [None] + self.dropout_layers

            if params.dropout > 0:
                self.dropout_skip_layers = [
                    tf.keras.layers.SpatialDropout2D(params.dropout)
                    ] + self.dropout_skip_layers
            else:
                self.dropout_skip_layers = [
                    None
                    ] + self.dropout_skip_layers

            if params.skip_connection_type == 'concat':
                num_ch_in_tmp = num_ch_in * 2  # x2 since channel is doubled after concatenation
            else:
                num_ch_in_tmp = num_ch_in
            if params.separable:
                layer.add(
                    SeparableTransposeConv2D(
                        filters=num_ch,
                        kernel_size=(params.kernel_size_time_de, params.kernel_size_freq),
                        activation=activation_layer,
                        num_channels_in=num_ch_in_tmp,  # x2 since channel is doubled after concatenation
                        batch_size=params.batchsize,
                        time_steps=params.time_steps,
                        normalization_layer=params.normalization_layer,
                        name=f"conv_tran_{i}"
                        ))
            else:
                if 0:
                    layer.add(
                        TransposeConv2D(
                            filters=num_ch,
                            kernel_size=(params.kernel_size_time, params.kernel_size_freq),
                            activation=params.activation,
                            name=f"conv_tran_{i}"
                            ))
                else:
                    
                    if activation_layer == 'glu':
                        num_ch_t = num_ch * 2
                    else:
                        num_ch_t = num_ch
                    layer.add(
                        tf.keras.layers.Conv2DTranspose(
                            filters=num_ch_t,
                            kernel_size=(params.kernel_size_time_de, params.kernel_size_freq),
                            strides=(1, 2),
                            padding='valid',
                            # activation=params.activation,
                            kernel_initializer='he_normal',
                            name=f"conv_tran_{i}"
                            ))
                    from soundkit.models.layers.activation_factory import ActivationFactory

                    act = ActivationFactory(activation_layer)

                    if isinstance(act.fn, tf.keras.layers.Layer):
                        # e.g., PReLU
                        layer.add(act.fn)
                    elif act.fn is None:
                        # no-op; skip adding an activation
                        pass
                    else:
                        # Wrap plain callables (relu/tanh/gelu/swish/softmax/2tanh/glu, etc.)
                        # Lambda makes it a proper Keras Layer for Sequential.add()
                        layer.add(tf.keras.layers.Lambda(
                            lambda x: act(x),
                            name=f"act_{params.activation or 'linear'}"))

                    layer.add(
                        SliceLayer(
                            params.kernel_size_time_de,
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

        for i, layer_info in enumerate(zip(inputs_dec[::-1], self.convs, self.states, self.zeros, self.dropout_layers, self.dropout_skip_layers)):

            encode, net, state, zeros, dropout, dropout_skip = layer_info

            state_en, state_de = state

            if self.params.kernel_size_time_de > 1:
                encode = tf.concat([state_en, encode], axis=1) # time concatenation
                x = tf.concat([state_de, x], axis=1) # time concatenation

                state_en_update=tf.identity(
                    encode[:,-(self.params.kernel_size_time_de-1):,:,:]
                    )

                state_de_update=tf.identity(
                    x[:,-(self.params.kernel_size_time_de-1):,:,:]
                    )

            if dropout is not None:
                x = dropout(x, training=training)

            if dropout_skip is not None:
                encode = dropout_skip(encode, training=training)
            if self.params.skip_connection_type == 'concat':
                comb = tf.concat([encode, x], axis=-1) # skip connection (channel concatenation)
            else:
                comb = encode + x # skip connection (channel addition)
            x = net(comb)
            # compensate the downsampling
            
            x = tf.concat([x, zeros], axis=2) if zeros is not None else x
            if self.params.kernel_size_time_de > 1:
                self.states[i][0].assign(state_en_update)
                self.states[i][1].assign(state_de_update)
        return x

    def make_states(self):
        """ Make states"""
        len_pad = self.params.kernel_size_time_de - 1

        states=[]
        for num_ch, freq_bin in zip(self.params.num_chs[1:], self.freq_bins[1:]):
            # reverse the order

            if self.params.kernel_size_time_de == 1:
                state_en = None
                state_de = None
            else:
                state_en = tf.zeros((self.params.batchsize, len_pad, freq_bin, num_ch))
                state_en = tf.Variable(state_en, trainable=False) # for eager mode
                state_de = tf.zeros((self.params.batchsize, len_pad, freq_bin, num_ch))
                state_de = tf.Variable(state_de, trainable=False) # for eager mode

            states = [(state_en, state_de)] + states

        return states

    def reset_states(self):
        """ Reset states"""
        if self.params.kernel_size_time_de > 1:
            for state in self.states:
                state[0].assign(state[0] * 0)
                state[1].assign(state[1] * 0)