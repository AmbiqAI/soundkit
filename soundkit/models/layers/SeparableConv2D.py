import tensorflow as tf
from .activation_factory import ActivationFactory
from .normalization_layer_factory import NormalizationFactory
class SeparableConv2D(tf.keras.layers.Layer):
    """ Separable convolutional layer"""
    def __init__(
            self,
            filters,
            kernel_size,
            strides=(1, 1),
            num_channels_in=1,
            activation=None,
            normalization_layer=None,
            depthwise_activation=None,
            **kwargs):
        super(SeparableConv2D, self).__init__(**kwargs)

        if normalization_layer in (None, 'None', 'none'):
            use_bias=True
        else:
            use_bias=False

        self.depthwise = tf.keras.layers.Conv2D(
            filters=num_channels_in,
            kernel_size = kernel_size,
            strides=strides,
            groups = num_channels_in,
            use_bias=False,
            kernel_initializer='he_normal',
            activation=None,)

        self.pointwise = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            use_bias=use_bias,
            kernel_initializer='he_normal',
            )

        self.norm = NormalizationFactory(normalization_layer)
        self.activation = ActivationFactory(activation)
        self.depthwise_activation = ActivationFactory(depthwise_activation)

    def call(self, inputs):
        """ Forward pass"""
        x = self.depthwise(inputs)
        x = self.depthwise_activation(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.activation(x)
        # tf.print("activation min:", tf.reduce_min(x))
        # tf.print("activation max:", tf.reduce_max(x))
        return x
