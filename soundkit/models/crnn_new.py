from pydantic import BaseModel
from typing import Optional, List
import tensorflow as tf
from ..utils.tf_basic_math import tf_log10_eps
import numpy as np
class CRNNParams(BaseModel):

    batchsize: int = 1
    time_steps: int = 1
    dim_feat: int = 257
    unroll_rnn: bool = False
    layer_configs: List[dict] = [
        {
            'type': 'fc',
            'units': 257,
            'activation': 'sigmoid'
        },
    ]

class CRNN(tf.keras.Model):
    """Convolutional Recurrent Neural Network (CRNN) for sequence prediction.
    This model consists of convolutional layers followed by a GRU layer and an output layer.
    It is designed for sequence prediction tasks.
    """
    def __init__(
            self,
            params: CRNNParams = CRNNParams(),
            **kwargs):
        super().__init__()
        self.params = params
        self.layers_list = []
        self.states = []
        self.stride_time=1

        # for speaker verification
        self.weight_cos = tf.Variable(30.0, dtype=tf.float32)
        self.bias_cos = tf.Variable(0, dtype=tf.float32)
        self.output_dims=[params.dim_feat]
        for i, layer_def in enumerate(params.layer_configs):
            # import pdb; pdb.set_trace()
            if layer_def['type'] == 'conv2d':
                self.output_dims.append(layer_def['filters'])
                self.layers_list.append(tf.keras.layers.Conv2D(
                    filters=layer_def['filters'],
                    kernel_size=layer_def['kernel_size'],
                    strides=layer_def['strides'],
                    activation=layer_def['activation'],
                    padding='valid',
                ))

                self.stride_time = layer_def['strides'][0]
            elif layer_def['type'] == 'lstm':
                self.output_dims.append(layer_def['units'])

                self.layers_list.append(tf.keras.layers.LSTM(
                    units=layer_def['units'],
                    return_sequences=True,
                    unroll=params.unroll_rnn,
                    stateful=False,
                    return_state=True,
                    unit_forget_bias = True,
                    activation='tanh',
                    recurrent_activation='sigmoid',
                ))

            elif layer_def['type'] == 'fc':
                self.output_dims.append(layer_def['units'])
                self.layers_list.append(tf.keras.layers.Dense(
                    units=layer_def['units'],
                    activation=layer_def['activation'],
                ))
            elif layer_def['type'] == 'batchnorm':
                self.output_dims.append(self.output_dims[-1])  # Keep the same output dimension as the previous layer

                self.layers_list.append(
                    tf.keras.layers.BatchNormalization(
                        momentum=layer_def['momentum'],
                        epsilon=float(layer_def['epsilon'])
                    ))
            elif layer_def['type']  == 'layernorm':
                self.output_dims.append(self.output_dims[-1])
                self.layers_list.append(
                    tf.keras.layers.LayerNormalization(
                    epsilon=float(layer_def['epsilon'])
                ))
            elif layer_def['type'] == 'dropout':
                self.output_dims.append(self.output_dims[-1])
                self.layers_list.append(tf.keras.layers.Dropout(rate=layer_def['rate']))
            else:
                raise ValueError(f"Unsupported layer type: {layer_def['type']}")
    
        # Initialize states for CNN and LSTM layers
        self.cnn_states = []
        cnn_idx = 0
        for i, layer_def in enumerate(params.layer_configs):
            if layer_def['type'] == 'conv2d':
                k_t = layer_def['kernel_size'][0]
                input_dim = self.output_dims[i]
                if k_t > 1:
                    state = tf.Variable(
                        tf.zeros((params.batchsize, k_t - 1, input_dim), dtype=tf.float32),
                        trainable=False,
                        # name=f"conv2d_state_{cnn_idx}"
                    )
                self.cnn_states.append(state)
                cnn_idx += 1

        self.h_states = []
        self.c_states = []
        h_idx = 0
        for i, layer_def in enumerate(params.layer_configs):
            if layer_def['type'] == 'lstm':
                h_state = tf.Variable(
                    tf.zeros((params.batchsize, layer_def['units']), dtype=tf.float32),
                    trainable=False,
                    name=f"h_state_{h_idx}",
                )
                c_state = tf.Variable(
                    tf.zeros((params.batchsize, layer_def['units']), dtype=tf.float32),
                    trainable=False,
                    name=f"c_state_{h_idx}",
                )
                self.h_states.append(h_state)
                self.c_states.append(c_state)
                h_idx += 1
        self.reset_states(zero_state=False)

    def reset_states(self, zero_state=False):

        """Reset the states of the CRNN model."""
        for state in self.cnn_states:
            state.assign(tf.zeros_like(state))

        for h_state in self.h_states:
            if zero_state:
                h_state.assign(tf.zeros_like(h_state))
            else:
                state = tf.random.truncated_normal(tf.shape(h_state), stddev=1/np.sqrt(tf.shape(h_state)[-1]))
                h_state.assign(tf.minimum(tf.maximum(state, -1.0), 1.0-2**-15))

        for c_state in self.c_states:
            if zero_state:
                c_state.assign(tf.zeros_like(c_state))
            else:
                c_state.assign(tf.random.truncated_normal(tf.shape(c_state), stddev=1/np.sqrt(tf.shape(c_state)[-1])))

    def call(self, x, mask = 1.0, reset_input=tf.constant([0.0], dtype=tf.float32), training=False):
        """Forward pass through the CRNN model."""
        
        reset_input = reset_input[0]

        idx_cnn = 0
        idx_lstm = 0

        for layer, config in zip(self.layers_list, self.params.layer_configs):
            if config['type'] == 'conv2d': # 1d conv
                k_t = config['kernel_size'][0]
                x = tf.concat([self.cnn_states[idx_cnn], x], axis=1)
                state_update= tf.identity(x[:,-(k_t-1):,:])
                x = tf.expand_dims(x, axis=-1)
                x = layer(x, training=training)
                x = x[:, :, 0, :]
                self.cnn_states[idx_cnn].assign(state_update * (1 - reset_input))  # Update state
                idx_cnn += 1
            elif config['type'] == 'lstm':
                h_state, c_state = self.h_states[idx_lstm], self.c_states[idx_lstm]
                x, h_state_update, c_state_update = layer(
                                x,
                                initial_state = (h_state, c_state),
                                training = training)
                h_state.assign(h_state_update * (1 - reset_input))
                c_state.assign(c_state_update * (1 - reset_input))
                idx_lstm += 1
            else:
                x = layer(x, training=training)
        return x
