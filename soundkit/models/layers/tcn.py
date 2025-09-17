''' Temporal Convolutional Network (TCN) Layer Implementation
A residual block and a TCN layer implementation in TensorFlow.
This code defines a residual block with two convolutional layers and a TCN layer that applies multiple
dilated convolutions. The residual block supports dropout and can be used in a TCN 
layer to process sequential data.
'''
import tensorflow as tf
from typing import List
from .activation_factory import ActivationFactory


class resnet_block(tf.keras.layers.Layer):
    """
    Residual block for ResNet architecture.
    This block applies two convolutional layers
    with ReLU activation.
    """
    def __init__(
            self,
            filters: int = 32,
            batchsize: int = 32,
            dilation: int = 2,
            kernel_size: int = 2,
            dim_feat: int = 100,
            dropout_rate: float = 0.0,
            second_dropout: bool = True,
            activation: str = 'relu',
            **kwargs):
        super(resnet_block, self).__init__(**kwargs)
        self.activation = activation
        self.activation_func = ActivationFactory(activation)
        self.dropout_rate = dropout_rate
        self.second_dropout = second_dropout
        self.kernel_size_prime= (kernel_size - 1) * dilation + 1
        self.batchsize = batchsize
        self.dim_feat = dim_feat
        self.filters = filters

        if activation == 'glu':
            filters_out = filters * 2
        else:
            filters_out = filters

        self.conv1 = tf.keras.layers.Conv1D(
            filters_out,
            kernel_size,
            dilation_rate=dilation,
            padding='valid')

        self.state1 = tf.Variable(
            initial_value=tf.zeros(
                (batchsize,
                 self.kernel_size_prime-1,
                 dim_feat)),
            trainable=False,
            name="state_resnet_1",
            dtype=tf.float32
        )

        self.conv2 = tf.keras.layers.Conv1D(
            filters_out,
            kernel_size,
            dilation_rate=dilation,
            padding='valid'
        )

        self.state2 = tf.Variable(
            initial_value=tf.zeros(
                (batchsize,
                 self.kernel_size_prime-1,
                 filters)),
            trainable=False,
            name="state_resnet_2",
            dtype=tf.float32
        )
        if dim_feat != filters:
            self.skip_conv = tf.keras.layers.Conv1D(
                filters,
                1,
                padding='valid'
            )
        else:
            self.skip_conv = None

         # ---- Dropout layers (respect training flag) ----
        if self.dropout_rate > 0:
            # Classic: apply between conv1 and conv2
            self.drop1 = tf.keras.layers.Dropout(self.dropout_rate)
            # Optional: also apply after conv2 (before residual add)
            self.drop2 = tf.keras.layers.Dropout(self.dropout_rate) if self.second_dropout else None
        else:
            self.drop1 = None
            self.drop2 = None

    def call(self, inputs, training=False):
        """Forward pass through the residual block."""
        skip = inputs  # Save the input for the skip connection
        if self.skip_conv is not None:
            skip = self.skip_conv(skip)

        x = tf.concat(   # Concatenate the state1 with the input
            [self.state1, inputs],
            axis=1)

        self.state1.assign(
            x[:, -(self.kernel_size_prime-1):, :]
            )  # Update state1 with the last part of x

        x = self.conv1(x)
        
        x = self.activation_func(x)

        if self.drop1 is not None:
            x = self.drop1(
                x, training=training)  # dropout only during training

        x = tf.concat(      # Concatenate the state2 with the input
            [self.state2, x],
            axis=1)

        self.state2.assign(
            x[:, -(self.kernel_size_prime-1):, :]
            )  # Update state2 with the last part of x

        x = self.conv2(x)

        if self.drop2 is not None:
            x = self.drop2(
                x, training=training) # dropout only during training

        if self.activation == 'glu':
            # Gated Linear Unit activation
            a, b = tf.split(x, num_or_size_splits=2, axis=-1)
            x = a * tf.sigmoid(b) + skip
            # x = (a + skip) * tf.sigmoid(b)
        else:
            x = self.activation_func(skip + x) # Add the skip connection

        return x

    def reset_states(self):
        """Reset the states of the residual block."""
        self.state1.assign(
            tf.zeros_like(self.state1))

        self.state2.assign(
            tf.zeros_like(self.state2))

class tcn(tf.keras.layers.Layer):
    """
    Temporal Convolutional Network (TCN) layer.
    This layer applies a series of dilated convolutions to the input data.
    """
    def __init__(
            self,
            filters: int = 32,
            kernel_size: int = 2,
            batchsize: int = 32,
            dim_feat: int = 100,
            dilations: List[int] = [1, 2, 4, 8, 16, 32],
            dropout_rate: float = 0.0,
            second_dropout: bool = False,
            activation: str = 'relu',
            **kwargs):

        super(tcn, self).__init__(**kwargs)

        self.filters = filters
        self.dilation_rate = dilations
        self.batchsize = batchsize
        self.layers= []
        self.states= []
        self.kernel_size = kernel_size
        for i, dilation in enumerate(self.dilation_rate):
            if i == 0:
                dim = dim_feat
            else:
                dim = self.filters

            self.layers.append(
                resnet_block(
                    filters=self.filters,
                    batchsize=self.batchsize,
                    dilation=dilation,
                    kernel_size=self.kernel_size,
                    dim_feat=dim,
                    dropout_rate=dropout_rate,
                    second_dropout=second_dropout,
                    name=f"tcn_layer_{i}",
                    activation=activation
                )
            )

    def call(self, inputs, training=False):
        """Forward pass through the TCN layer."""
        x = inputs

        for layer in self.layers:
            x = layer(
                x, training=training)

        return x

    def reset_states(self):
        """Reset the states of the TCN layers."""
        for layer in self.layers:
            layer.reset_states()

if __name__ == "__main__":
    # Example usage
    dim_feat = 200
    tcn_layer = tcn(filters=32, kernel_size=2, dim_feat=dim_feat, batchsize=32)
    input_data = tf.random.normal((32, 1500, dim_feat))  # Batch size of 32, sequence length of 1500, 100 features
    output_data = tcn_layer(input_data)
    print(output_data.shape)  # Should print (32, 1500, 32) if filters=32