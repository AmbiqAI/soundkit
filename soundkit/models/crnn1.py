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
        self.rv=tf.Variable(
            0,
             trainable=False,
              dtype=tf.float32)  # Random variable for state initialization
        self.dense = tf.keras.layers.Dense(
            units=3
        )
    def reset_states(self, zero_state=False):

        self.rv.assign(1 )

    def call(self, x, mask = 1.0, training=False):
        """Forward pass through the CRNN model."""
        y = self.dense(x)
        y = y[...,-1] + self.rv
        self.rv.assign(self.rv + 1)
        return y
