import tensorflow as tf
def glu(x, axis=-1):
    a, b = tf.split(x, num_or_size_splits=2, axis=axis)
    return a * tf.sigmoid(b)
class ActivationFactory:
    def __init__(self, name: str | None):
        name = name.lower() if name else None

        if name == "relu":
            self.fn = tf.nn.relu
        elif name == "tanh":
            self.fn = tf.nn.tanh
        elif name == "2tanh":
            self.fn = lambda x: 2.0 * tf.nn.tanh(x)
        elif name == "sigmoid":
            self.fn = tf.nn.sigmoid
        elif name == "leaky_relu":
            self.fn = tf.nn.leaky_relu
        elif name == "gelu":
            self.fn = tf.nn.gelu
        elif name == "softmax":
            self.fn = tf.nn.softmax
        elif name == "softsign":
            self.fn = tf.nn.softsign
        elif name == "softplus":
            self.fn = tf.nn.softplus
        elif name == "elu":
            self.fn = tf.nn.elu
        elif name == "selu":
            self.fn = tf.nn.selu
        elif name == "swish":
            self.fn = tf.nn.swish
        elif name == "prelu":
            # PReLU is a layer, not a function, and needs to be built with input shape
            self.fn = tf.keras.layers.PReLU()
        elif name == "linear":
            self.fn = None
        elif name == "glu":
            self.fn = lambda x: glu(x)  # default axis=-1
        elif name in (None, "None", "none"):
            self.fn = None
        else:
            raise ValueError(f"Unsupported activation: '{name}'")

    def __call__(self, x):
        if self.fn is None:
            return x
        return self.fn(x)
