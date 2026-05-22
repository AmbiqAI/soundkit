import tensorflow as tf


class DyT(tf.keras.layers.Layer):
    """Dynamic Tanh (DyT) layer: gamma * tanh(alpha * x) + beta.
    
    Args:
        alpha_init: Initial value for the learnable alpha scalar.
        trainable_beta: If True, beta is a trainable parameter; if False, beta is fixed at zero.
    """
    def __init__(self, alpha_init=0.5, trainable_beta=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha_init = alpha_init
        self.trainable_beta = trainable_beta

    def build(self, input_shape):
        dim = input_shape[-1]
        self.alpha = self.add_weight(
            "alpha", shape=(),
            initializer=tf.constant_initializer(self.alpha_init),
            trainable=True)
        self.gamma = self.add_weight(
            "gamma", shape=(dim,),
            initializer="ones",
            trainable=True)
        self.beta = self.add_weight(
            "beta", shape=(dim,),
            initializer="zeros",
            trainable=self.trainable_beta)

    def call(self, x):
        alpha = tf.nn.softplus(self.alpha)
        gamma = tf.nn.softplus(self.gamma)
        return gamma * tf.math.tanh(alpha * x) + self.beta


class ActivationFactory(tf.keras.layers.Layer):
    def __init__(self, name: str | None, **kwargs):
        # Handle the case where 'name' is already an ActivationFactory instance
        if hasattr(name, '_act_name'):
            name = name._act_name

        act_name = name.lower() if name else None
        super().__init__(name=f"act_{act_name}" if act_name else "act_linear", **kwargs)
        self._act_name = act_name
        self._sublayer = None

        if act_name == "relu":
            self.fn = tf.nn.relu
        elif act_name == "relu6":
            self.fn = tf.nn.relu6
        elif act_name == "hard_tanh128":
            scale = 128.0 / 1.0
            def hard_tanh128(x):
                out = tf.clip_by_value(x / scale, -1.0, 1.0) * scale
                return out
            self.fn = lambda x: hard_tanh128(x)
        elif act_name == "relu256":
            scale = 256.0 / 6.0
            def relu256(x):
                out = tf.nn.relu6(x / scale) * scale
                return out
            self.fn = lambda x: relu256(x)
        elif act_name == "tanh":
            self.fn = tf.nn.tanh
        elif act_name == "sigmoid":
            self.fn = tf.nn.sigmoid
        elif act_name == "leaky_relu":
            self.fn = tf.nn.leaky_relu
        elif act_name == "gelu":
            self.fn = tf.nn.gelu
        elif act_name == "softmax":
            self.fn = tf.nn.softmax
        elif act_name == "softplus":
            self.fn = tf.nn.softplus
        elif act_name == "elu":
            self.fn = tf.nn.elu
        elif act_name == "selu":
            self.fn = tf.nn.selu
        elif act_name == "swish":
            self.fn = tf.nn.swish
        elif act_name == "prelu":
            self._sublayer = tf.keras.layers.PReLU(shared_axes=[1, 2])
            self.fn = None
        elif act_name == "dyt":
            self._sublayer = DyT(alpha_init=0.5, trainable_beta=False)
            self.fn = None
        elif act_name in (None, "none", "linear"):
            self.fn = None
        else:
            raise ValueError(f"Unsupported activation: '{act_name}'")

    def call(self, x):
        if self._sublayer is not None:
            return self._sublayer(x)
        if self.fn is None:
            return x
        return self.fn(x)