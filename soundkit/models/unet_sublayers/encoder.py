import tensorflow as tf
from . import UNetParams
from . import UNetParams, get_unet_info
from ..layers.SeparableConv2D import SeparableConv2D

class encoder_unet(tf.keras.layers.Layer):
    """ Encoder of UNet"""
    def __init__(
            self,
            params: UNetParams = UNetParams(),
            **kwargs):
        super(encoder_unet, self).__init__(**kwargs)

        self.params = params
        self.convs=[]

        self.freq_bins,_ = get_unet_info(
            params.num_chs,
            dim_feat=params.dim_feat)
        # self.freq_bins = [257, 128, 63, 31, 15]
        stages = len(params.num_chs) - 1
        self.states = self.make_states()
        for i , num_ch, num_ch_in in zip(range(stages), params.num_chs[1:], params.num_chs[:-1]):

            layer=tf.keras.Sequential(name=f"encoder_{i}")
            if params.dropout > 0:
                drop_rate=0.1 if i == 0 else params.dropout
                layer.add(
                    tf.keras.layers.Dropout(
                        rate=drop_rate,
                        name=f"dropout_{i}"
                        ))

            if params.separable:
                layer.add(
                    SeparableConv2D(
                        filters=num_ch,
                        kernel_size=(params.kernel_size_time, 3),
                        strides=(1, 2),
                        activation=params.activation,
                        num_channels_in=num_ch_in,
                        normalization_layer=params.normalization_layer,
                        name=f"conv_{i}"
                        ))
            else:
                layer.add(
                    tf.keras.layers.Conv2D(
                        filters=num_ch,
                        kernel_size=(params.kernel_size_time, 3),
                        strides=(1, 2),
                        padding='valid',
                        activation=params.activation,
                        kernel_initializer='he_normal',
                        name=f"conv_{i}"
                        ))

            self.convs += [layer]

    def make_states(self):
        """ Make states"""
        states = []

        len_pad = self.params.kernel_size_time - 1

        for i, da in enumerate(zip(self.params.num_chs[:-1], self.freq_bins[:-1])):
            num_ch, freq_bin = da
            shape = (self.params.batchsize, len_pad, freq_bin, num_ch)

            state=tf.zeros(shape)
            state = tf.Variable(state, trainable=False) # for eager mode
            states += [state]

        return states

    def reset_states(
            self):
        """ Reset states"""

        len_pad = self.params.kernel_size_time - 1

        for i, da in enumerate(zip(self.params.num_chs[:-1], self.freq_bins[:-1])):
            num_ch, freq_bin = da
            shape = (self.params.batchsize, len_pad, freq_bin, num_ch)

            state=tf.zeros(shape)
            state = tf.Variable(state, trainable=False) # for eager mode
            self.states[i].assign(state)

    def call(
            self,
            inputs,
            # states,
            training=False):
        """ Forward pass"""

        x = inputs
        outputs= []

        for i, layer_info in enumerate(zip(self.states, self.convs)):
            state, net = layer_info
            x = tf.concat([state, x], axis=1)
            state_update=tf.identity(x[:,-(self.params.kernel_size_time-1):,:,:])
            x = net(x)
            self.states[i].assign(state_update)
            outputs+= [x]

        # self.states = states_udpate
        return outputs