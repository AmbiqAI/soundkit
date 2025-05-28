import tensorflow as tf

class ActivationFactory:
    def __init__(self, name: str | None):
        name = name.lower() if name else None

        if name == "relu":
            self.fn = tf.nn.relu
        elif name == "tanh":
            self.fn = tf.nn.tanh
        elif name == "sigmoid":
            self.fn = tf.nn.sigmoid
        elif name == "leaky_relu":
            self.fn = tf.nn.leaky_relu
        elif name == "gelu":
            self.fn = tf.nn.gelu
        elif name == "softmax":
            self.fn = tf.nn.softmax
        elif name == "softplus":
            self.fn = tf.nn.softplus
        elif name == "elu":
            self.fn = tf.nn.elu
        elif name == "selu":
            self.fn = tf.nn.selu
        elif name == "swish":
            self.fn = tf.nn.swish
        elif name in (None, "None", "none"):
            self.fn = None
        else:
            raise ValueError(f"Unsupported activation: '{name}'")

    def __call__(self, x):
        if self.fn is None:
            return x
        return self.fn(x)
