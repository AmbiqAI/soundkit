import tensorflow as tf

class ActivationFactory:
    def __init__(self, name: str | None):
        # Handle the case where 'name' is already an ActivationFactory instance
        # (Fixing the double-wrapping bug from earlier)
        if hasattr(name, 'fn'):
            self.fn = name.fn
            return

        name = name.lower() if name else None

        if name == "relu":
            self.fn = tf.nn.relu
        elif name == "relu6":
            self.fn = tf.nn.relu6
        elif name == "relu256":
            # ReLU256 is not a built-in TF function, so we define it manually.
            # We use clip_by_value(0, 256) which is equivalent to min(max(x, 0), 256)

            def relu256(x):
                out = tf.maximum(tf.minimum(x, 256.0), 0.0)
                d = 256 / 32767.0
                # out = tf.floor(out / d) * d
                return out
            self.fn = lambda x: relu256(x)

        # === Added clip256 ===
        elif name == "clip256":
            # clip256 is not a built-in TF function, so we define it manually.
            # We use clip_by_value(0, 256) which is equivalent to min(max(x, 0), 256)
            def clip256(x):
                out = tf.maximum(tf.minimum(x, 256.0), -256.0)
                d = 512 / 65535.0
                # out = tf.floor(out / d) * d
                return out
            self.fn = lambda x: clip256(x)
        
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
        elif name in (None, "None", "none", "linear"):
            self.fn = None
        else:
            raise ValueError(f"Unsupported activation: '{name}'")

    def __call__(self, x, **kwargs):
        if self.fn is None:
            return x
        return self.fn(x)