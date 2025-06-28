import tensorflow as tf
from pydantic import BaseModel
from typing import Optional, List

class newCRNNParams(BaseModel):

    dim_feat: int = 257
    stride_time: int = 1
    unroll_rnn: bool = False
    layer_configs: List[dict] = [
        {
            'type': 'fc',
            'units': 257,
            'activation': 'sigmoid'
        },
    ]


class crnn_new(tf.keras.Model):
    def __init__(
            self,
            params: newCRNNParams = newCRNNParams(),
            **kwargs):
        super(crnn_new, self).__init__()

        self.dim_feat = params.dim_feat
        self.layer_configs = params.layer_configs
        self.unroll_rnn = params.unroll_rnn
        self.stride_time = params.stride_time

        self.layers_list = []
        self.cnn_states = []
        self.h_states = []
        self.c_states = []
        self.lstm_units = []
        self.output_dims = [self.dim_feat]  # Track output dims per layer for cnn state shapes
        
        # for speaker verification

        self.weight_cos = tf.Variable(30.0, dtype=tf.float32)
        self.bias_cos = tf.Variable(0, dtype=tf.float32)

        for idx, layer_def in enumerate(self.layer_configs):
            layer_type = layer_def["type"]
            activation = layer_def.get("activation", "linear")

            if layer_type == "conv2d":
                filters = layer_def["filters"]
                kernel_size = layer_def["kernel_size"]
                self.layers_list.append(tf.keras.layers.Conv2D(
                    filters=filters,
                    strides=layer_def.get("strides", (1, 1)),
                    kernel_size=kernel_size,
                    activation=activation,
                    padding='valid'
                ))
                self.cnn_states.append(None)
                self.output_dims.append(filters)

            elif layer_type == "lstm":
                units = layer_def["units"]
                self.layers_list.append(tf.keras.layers.LSTM(
                    units=units,
                    return_sequences=True,
                    return_state=True,
                    unroll=self.unroll_rnn,
                    activation=activation,
                    recurrent_activation='sigmoid'
                ))
                self.h_states.append(None)
                self.c_states.append(None)
                self.lstm_units.append(units)
                self.output_dims.append(units)

            elif layer_type == "fc":
                units = layer_def["units"]
                self.layers_list.append(tf.keras.layers.Dense(
                    units=units,
                    activation=activation
                ))
                self.output_dims.append(units)

            elif layer_type == "batchnorm":
                self.layers_list.append(tf.keras.layers.BatchNormalization())
                self.output_dims.append(self.output_dims[-1])

            elif layer_type == "layernorm":
                self.layers_list.append(tf.keras.layers.LayerNormalization())
                self.output_dims.append(self.output_dims[-1])

            elif layer_type == "dropout":
                rate = layer_def.get("rate", 0.5)
                self.layers_list.append(tf.keras.layers.Dropout(rate))
                self.output_dims.append(self.output_dims[-1])

            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

    def build(self, input_shape):
        # super(crnn_new, self).build(input_shape)
        batch_size = input_shape[0]
        cnn_idx = 0
        for idx, layer_def in enumerate(self.layer_configs):
            if layer_def["type"] == "conv2d":

                k_t = layer_def["kernel_size"][0]
                feature_dim = self.output_dims[idx]

                self.cnn_states[cnn_idx] = self.add_weight(
                    shape=(batch_size, k_t - 1, feature_dim),
                    trainable=False,
                    name=f"cnn_state_{cnn_idx}",
                    initializer="zeros"
                )

                cnn_idx += 1

        h_idx = 0
        for idx, layer_def in enumerate(self.layer_configs):
            if layer_def["type"] == "lstm":
                units = layer_def["units"]
                self.h_states[h_idx] = self.add_weight(
                    shape=(batch_size, units),
                    trainable=False,
                    name=f"h_state_{h_idx}",
                    initializer="zeros"
                )

                self.c_states[h_idx] = self.add_weight(
                    shape=(batch_size, units),
                    trainable=False,
                    name=f"c_state_{h_idx}",
                    initializer="zeros"
                )


                h_idx += 1

    def reset_states(self, zero_state=True):
        for state in self.cnn_states + self.h_states + self.c_states:
            if state is not None:
                state.assign(tf.zeros_like(state))

    def call(self, x_input,mask = 1.0, reset_input=[0.0], training=False):
        cnn_idx = 0
        lstm_idx = 0
        reset_input = reset_input[0]
        for layer, layer_def in zip(self.layers_list, self.layer_configs):
            layer_type = layer_def["type"]

            if layer_type == "conv2d":
                
                k_t = layer_def["kernel_size"][0]
                x_input = tf.concat([self.cnn_states[cnn_idx], x_input], axis=1)
                new_state = x_input[:, -k_t + 1:, :]

                self.cnn_states[cnn_idx].assign(new_state * (1 - reset_input))
                x_input = tf.expand_dims(x_input, axis=-1)  # [B, T, D, 1]
                x_input = layer(x_input)
                x_input = tf.squeeze(x_input, axis=-2)
                cnn_idx += 1

            elif layer_type == "lstm":
                x_input, h_new, c_new = layer(x_input, initial_state=[self.h_states[lstm_idx], self.c_states[lstm_idx]])
                self.h_states[lstm_idx].assign(h_new * (1 - reset_input))
                self.c_states[lstm_idx].assign(c_new * (1 - reset_input))
                lstm_idx += 1

            else:
                x_input = layer(x_input, training=training) if layer_type == "dropout" else layer(x_input)

        return x_input

if __name__ == "__main__":

    params = newCRNNParams(
        dim_feat=257,
        unroll_rnn=True,
        layer_configs=[
            {'type': 'conv2d', 'filters': 32, 'kernel_size': (3, 257), 'activation': 'relu'},
            {'type': 'lstm', 'units': 64, 'activation': 'tanh'},
            {'type': 'fc', 'units': 10, 'activation': 'softmax'}
        ]
    )
    model = crnn_new(params)
    inputs_x = tf.keras.Input(shape=[100, 257], batch_size=3, dtype=tf.float32, name="x_input")
    inputs_reset = tf.keras.Input(shape=(), batch_size=3, dtype=tf.float32, name="reset_input")
    model(inputs_x, reset_input=inputs_reset)

    for v in model.trainable_variables:
        print(f"{v.name}: {v.shape}")
    # print(model.summary())

    model.save_weights("trained_model_weights.h5")
    
    model_load = crnn_new(params)
    inputs_x = tf.keras.Input(shape=[100, 257], batch_size=1, dtype=tf.float32, name="x_input")
    inputs_reset = tf.keras.Input(shape=(), batch_size=1, dtype=tf.float32, name="reset_input")
    model_load(inputs_x, reset_input=inputs_reset)
    model_load.load_weights("trained_model_weights.h5", by_name=True, skip_mismatch=True)
    for v in model.trainable_variables:
        print(f"{v.name}: {v.shape}")
    # print(model.summary())

    import pdb; pdb.set_trace()
    
    
    # import pdb; pdb.set_trace()
    # inputs_x = tf.random.normal(shape=(1, 100, 257), dtype=tf.float32)
    # inputs_reset = tf.constant([0.0], dtype=tf.float32)
    # model(inputs_x, reset_input=inputs_reset)
    # for v in model.trainable_variables:
    #     print(f"{v.name}: {v.shape}")

    # print(model.summary())