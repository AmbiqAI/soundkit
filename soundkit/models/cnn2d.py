import tensorflow as tf
from pydantic import BaseModel
from typing import Type, Dict, List
from .layers.activation_factory import ActivationFactory
from .layers.normalization_layer_factory import NormalizationFactory
from .layers.tcn import tcn

class CNN2DParams(BaseModel):

    batchsize: int = 1
    time_steps: int = 1
    dim_feat: int = 480
    unroll_rnn: bool = False
    activation: str = 'glu'
    kernel_size: List[int] = [1, 5]
    channels: List[int] = [1, 8, 8, 8, 8]
    dim_out_rnn: int = 64
    dim_out: int = 2
    dropout: float = 0.0
    normalization_layer: str | None = None
    rnn_res: bool = False
    kernel_sizes: List[List[int]] = [[1, 5], [1, 5], [1, 5], [1, 5]]
    time_net: str = 'lstm'
    dilations: List[int] = [1, 2, 4, 8, 16, 32]
    filters_tcn: int = 64
    kernel_size_tcn: int = 2

class CNN2D(tf.keras.Model):
    """Convolutional Recurrent Neural Network (CRNN) for sequence prediction.
    This model consists of convolutional layers followed by a LSTM layer and an output layer.
    It is designed for sequence prediction tasks.
    """
    def __init__(
            self,
            params: CNN2DParams = CNN2DParams(),
            **kwargs):
        super(CNN2D, self).__init__(**kwargs)

        self.params = params

        self.states = []
        self.cnn_layers = []

        batchsize=params.batchsize

        channels=params.channels
        dropout=params.dropout

        activation = params.activation

        dim = params.dim_feat

        for i, ch_info in enumerate(zip(channels[:-1], channels[1:], params.kernel_sizes)):

            ch_in, ch_out, kernel_size = ch_info

            filter_size_time, filter_size_freq = kernel_size

            layer = tf.keras.Sequential(name=f"cnn_layer_{i}")
            
            # Create dropout layer if dropout is specified            
            if dropout > 0:
                if i == 0:
                    dropout_prob = 0.0
                    dropout_layer = tf.keras.layers.Dropout(
                        rate=dropout_prob,
                        name=f"dropout_{i}"
                    )
                else:
                    dropout_prob = dropout

                    dropout_layer = tf.keras.layers.SpatialDropout2D(
                        rate=dropout_prob,
                        name=f"dropout_{i}"
                    )

                layer.add(dropout_layer)
            if (dim - filter_size_freq) // 2 == 1:
                pad_1 = True
                dim += 1
            else:
                pad_1 = False

            if pad_1:
                layer.add(
                    tf.keras.layers.ZeroPadding2D(
                        padding=((0, 0), (1, 0)),
                        name=f"pad_{i}"
                    )
                )
            dim = 1 + ((dim - filter_size_freq) // 2)

            if filter_size_time == 1:
                state = None
            else:
                state = tf.Variable(
                    initial_value=tf.zeros(
                        (batchsize, filter_size_time-1, dim, ch_in)),
                    trainable=False,
                    name=f"state_{i}",
                    dtype=tf.float32
                )

            self.states.append(state)

            if activation in ('relu', 'relu6', 'glu'):
                kernel_initializer = 'he_normal'
            else:
                kernel_initializer = 'glorot_uniform'

            if activation == 'glu':
                ch_out *= 2  # GLU doubles the output channels

            layer.add(
                tf.keras.layers.Conv2D(
                    filters=ch_out,
                    kernel_size=kernel_size,
                    strides=(1, 2),
                    padding='valid',
                    kernel_initializer=kernel_initializer,
                    name=f"conv_{i}"
                )
            )

            if params.normalization_layer is not None:
                layer.add(
                    NormalizationFactory(params.normalization_layer)
                )

            layer.add(ActivationFactory(activation))

            self.cnn_layers.append(layer)

        
        if dropout > 0:
            self.cnn_layers.append(
                tf.keras.layers.SpatialDropout2D(
                    rate=dropout,
                    name=f'cnn_dropout'
                )
            )

        self.len_time = 50  # Length of time dimension for CNN
        if params.time_net == 'cnn':

            self.layer_time = tf.keras.layers.Conv2D(
                        filters=ch_out,
                        kernel_size=(self.len_time, 1),
                        strides=(1, 1),
                        padding='valid',
                        kernel_initializer=kernel_initializer,
                        name=f"conv_{i}"
                    )
            self.layer_time_act = ActivationFactory(activation)
            self.state_time = tf.Variable(
                initial_value=tf.zeros(
                    (batchsize, self.len_time-1, dim, ch_out)),
                trainable=False,
                name=f"state_time",
                dtype=tf.float32
            )
        elif params.time_net == 'tcn':
            self.layer_time = tcn(
                filters         = params.filters_tcn,
                kernel_size     = params.kernel_size_tcn,
                batchsize       = batchsize,
                dim_feat        = dim * ch_out,
                dilations       = params.dilations,
                dropout_rate    = dropout,
                second_dropout  = True,
                activation=params.activation  ,
                name            = "tcn_layer"
            )

        elif params.time_net == 'lstm':
            self.rnn_h = tf.Variable(
                tf.zeros(
                    (batchsize, params.dim_out_rnn), dtype=tf.float32),
                trainable=False,
                # name="rnn_h"
            )

            self.rnn_c = tf.Variable(
                tf.zeros((batchsize, params.dim_out_rnn), dtype=tf.float32),
                trainable=False,
                # name="rnn_c"
            )

            self.layer_time = tf.keras.layers.LSTM(
                    units=params.dim_out_rnn,
                    return_sequences=True,
                    unroll=params.unroll_rnn,
                    stateful=False,
                    return_state=True,
                    unit_forget_bias = True,
                    activation='tanh',
                    recurrent_activation='sigmoid',
                )

        # Output layer
        self.fc = tf.keras.Sequential(name="output_layer")
        self.fc.add(
            tf.keras.layers.Dense(
                units=params.dim_out,
                name='output_dense'
            )
        )
        self.reset_states()

    def reset_states(self):
        """ Reset states"""
        for i, state in enumerate(self.states):
            if state is not None:
                # Reset the state to zeros
                state.assign(tf.zeros_like(state))
        if self.params.time_net == 'lstm':
            h = tf.random.uniform(
                shape=self.rnn_h.shape,
                minval=-1.0,
                maxval=1.0,
                dtype=tf.float32,)

            c = tf.random.truncated_normal(
                shape=self.rnn_c.shape,
                mean=0.0,
                stddev=1.0,
                dtype=tf.float32)
            self.rnn_h.assign(h)
            self.rnn_c.assign(c)
        elif self.params.time_net == 'cnn':
            self.state_time.assign(
                tf.zeros_like(self.state_time)
            )
        elif self.params.time_net == 'tcn':
            self.layer_time.reset_states()

    def call(
            self,
            inputs,
            mask=1.0,
            # reset_input=tf.constant([0.0], dtype=tf.float32),
            training=False):
        """ Forward pass"""
        # reset_input = reset_input[0]
        if len(inputs.shape) == 3:
            # (B, T, F) → (B, T, F, 1)
            x = tf.expand_dims(inputs, axis=-1)
        else:
            x = inputs

        for i, (kernel_size, layer, state) in enumerate(zip(self.params.kernel_sizes, self.cnn_layers, self.states)):
            filter_size_time, _ = kernel_size
            if state is not None:
                # Concatenate the state to the input
                x_concat = tf.concat([state, x], axis=1)
                new_state = x_concat[:, -(filter_size_time - 1):, :, :]  # Save last (k-1) steps
                self.states[i].assign(new_state)
            else:
                x_concat = x

            x = layer(x_concat, training=training)


        if self.params.time_net == 'lstm':
            timesteps = tf.shape(x)[1]
            
            x = tf.reshape(x, (self.params.batchsize, timesteps, -1))

            x, h, c = self.layer_time(
                x,
                initial_state=(self.rnn_h, self.rnn_c),
                training=training
            )
            self.rnn_h.assign(h)
            self.rnn_c.assign(c)

        elif self.params.time_net == 'tcn':
            timesteps = tf.shape(x)[1]
            x = tf.reshape(x, (self.params.batchsize, timesteps, -1))

            x = self.layer_time(
                x,
                training=training
            )

        elif self.params.time_net == 'cnn':
            x = tf.concat([self.state_time, x], axis=1)
            self.state_time.assign(
                x[:, -(self.len_time - 1):, :, :])

            x = self.layer_time(x, training=training)
            x = self.layer_time_act(x)

            timesteps = tf.shape(x)[1]
            x = tf.reshape(x, (self.params.batchsize, timesteps, -1))

        x = self.fc(x, training=training)

        return x

if __name__ == "__main__":
    # Example usage
    model = crnn2d()
    inputs = tf.random.normal((32, 160*1500))  # Batch of 32 samples, each with 160 time steps and 1 channel
    outputs = model(inputs)
    print(outputs.shape)  # Should print the shape of the output tensor