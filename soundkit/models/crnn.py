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
        self.layer_stack = []
        self.states = []
        self.stride_time=1


        for i, layer_def in enumerate(params.layer_configs):
            if layer_def['type'] == 'conv2d':
                units_in = self.get_former_neurons(params, i)

                state = tf.Variable(tf.zeros(
                        (params.batchsize,
                        layer_def['kernel_size'][0] - 1,
                        units_in,),
                        dtype=tf.float32),
                        trainable=False)  # type: ignore
               
                
                self.states.append(
                        state
                    )
                
                self.layer_stack.append(tf.keras.layers.Conv2D(
                    filters=layer_def['filters'],
                    kernel_size=layer_def['kernel_size'],
                    strides=layer_def['strides'],
                    activation=layer_def['activation'],
                    padding='valid'
                ))

                self.stride_time = layer_def['strides'][0]
            elif layer_def['type'] == 'lstm':

                h_states = tf.Variable(
                            tf.random.uniform(
                                [params.batchsize, layer_def['units']],
                                minval = -1.0,
                                maxval = 1.0),
                            dtype = tf.float32,
                            trainable = False)
                c_states = tf.Variable(
                            tf.random.truncated_normal([params.batchsize, layer_def['units']]),
                            dtype = tf.float32,
                            trainable = False)

                self.states.append([h_states, c_states])

                self.layer_stack.append(tf.keras.layers.LSTM(
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
                self.states.append(tf.constant(0, dtype=tf.float32))
                self.layer_stack.append(tf.keras.layers.Dense(
                    units=layer_def['units'],
                    activation=layer_def['activation']
                ))
            elif layer_def['type'] == 'batchnorm':
                self.states.append(tf.constant(0, dtype=tf.float32))

                self.layer_stack.append(
                    tf.keras.layers.BatchNormalization(
                        momentum=layer_def['momentum'],
                        epsilon=float(layer_def['epsilon'])
                    ))
            elif layer_def['type']  == 'layernorm':
                self.states.append(tf.constant(0, dtype=tf.float32))

                self.layer_stack.append(
                    tf.keras.layers.LayerNormalization(
                    epsilon=float(layer_def['epsilon'])
                ))
            elif layer_def['type'] == 'dropout':
                self.states.append(tf.constant(0, dtype=tf.float32))
                self.layer_stack.append(tf.keras.layers.Dropout(rate=layer_def['rate']))
            else:
                raise ValueError(f"Unsupported layer type: {layer_def['type']}")

    def call(self, x, mask = 1.0, training=False):
        """Forward pass through the CRNN model."""
        for layer, config, state in zip(self.layer_stack, self.params.layer_configs, self.states):
            if config['type'] == 'conv2d':
                
                x = tf.concat([state, x], axis=1)
                state_update= tf.identity(x[:,-(config['kernel_size'][0]-1):,:])

                x = tf.expand_dims(x, axis=-1)
                
                
                x = layer(x, training=training)
                x = x[:, :, 0, :]
                state.assign(state_update)
            elif config['type'] == 'lstm':
                h_state, c_state = state
                x, h_state_update, c_state_update = layer(
                                x,
                                initial_state = (h_state, c_state),
                                training = training)
                h_state.assign(h_state_update)
                c_state.assign(c_state_update)
            else:
                x = layer(x, training=training)
        return x
    def get_former_neurons(self, params, current_layer_index):
        """Get the number of input features for the current layer."""

        if current_layer_index == 0:
            return params.dim_feat
        else:
            
            for layer_config in reversed(params.layer_configs[:current_layer_index]):
                if layer_config['type'] in ['lstm', 'fc']:
                    num = layer_config['units']
                    break
                elif layer_config['type'] == 'conv2d':
                    num = layer_config['filters']
                    break
                else:
                    num=-1
            if num == -1:
                num = params.dim_feat
            
            return num