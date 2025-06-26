from pydantic import BaseModel
from typing import Optional, List
import tensorflow as tf
from ..utils.tf_basic_math import tf_log10_eps
import numpy as np

class ConvLSTMHybridParams(BaseModel):

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

class ConvLSTMHybridModel(tf.Module):
    def __init__(self,
                 params: ConvLSTMHybridParams = ConvLSTMHybridParams(),
                 **kwargs):
        super().__init__()
        self.params = params
        
        self.conv = tf.keras.layers.Conv2D(filters=100, kernel_size=(6, 40), padding="valid", activation='tanh')
        self.lstm = tf.keras.layers.LSTM(
                    units=100,
                    return_sequences=True,
                    unroll=params.unroll_rnn,
                    stateful=False,
                    return_state=True,
                    unit_forget_bias=True,
                    activation='tanh',
                    recurrent_activation='sigmoid'
                )
        self.dense = tf.keras.layers.Dense(64, activation='tanh')

    @tf.function(input_signature=[
        tf.TensorSpec([None, 180, 40], tf.float32, name="input_0"),     # main input (e.g. image/feature)
        tf.TensorSpec([None, 5, 40], tf.float32, name="input_1"),     # concat input (e.g. state feature)
        tf.TensorSpec([None, 100], tf.float32, name="input_2"),           # h_state
        tf.TensorSpec([None, 100], tf.float32, name="input_3"),           # c_state
    ])
    def __call__(self, input_0, input_cnn, input_h, input_c):
        # 1. Concatenate input and state along channels

        x = tf.concat([input_0, input_cnn], axis=-1)  # shape: [batch, 8, 8, 4]
        state_update = x[:, -5:, :]
        x = tf.expand_dims(x, axis=-1)
        x = self.conv(x)
        x = tf.squeeze(x, axis=-2)

        # 3. Flatten and reshape for LSTM
        
        # 4. LSTM with state
        lstm_out, h, c = self.lstm(x, initial_state=[input_h, input_c])

        # 5. Final dense layer
        output = self.dense(lstm_out)  # [batch, 1]

        return {
            "output": output,
            "new_input_cnn": state_update,
            "new_h": h,
            "new_c": c,
        }
