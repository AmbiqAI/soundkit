import math

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
        self._fused = False

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

    def fuse(self):
        """Precompute softplus(alpha) and softplus(gamma) into constants.
        After fuse(), call() uses direct multiply without softplus."""
        self.alpha.assign(tf.nn.softplus(self.alpha))
        self.gamma.assign(tf.nn.softplus(self.gamma))
        self._fused = True

    def call(self, x):
        if self._fused:
            return self.gamma * tf.math.tanh(self.alpha * x) + self.beta
        alpha = tf.nn.softplus(self.alpha)
        gamma = tf.nn.softplus(self.gamma)
        return gamma * tf.math.tanh(alpha * x) + self.beta


class DyS(tf.keras.layers.Layer):
    """Dynamic Sigmoid (DyS) layer: gamma * sigmoid(alpha * x) + beta.

    Non-negative output suitable for masking.
    
    Args:
        alpha_init: Initial value for the learnable alpha scalar.
        trainable_beta: If True, beta is a trainable parameter; if False, beta is fixed at zero.
    """
    def __init__(self, alpha_init=0.5, trainable_beta=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha_init = alpha_init
        self.trainable_beta = trainable_beta
        self._fused = False

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

    def fuse(self):
        """Precompute softplus(alpha) and softplus(gamma) into constants.
        After fuse(), call() uses direct multiply without softplus."""
        self.alpha.assign(tf.nn.softplus(self.alpha))
        self.gamma.assign(tf.nn.softplus(self.gamma))
        self._fused = True

    def call(self, x):
        if self._fused:
            return self.gamma * tf.math.sigmoid(self.alpha * x) + self.beta
        alpha = tf.nn.softplus(self.alpha)
        gamma = tf.nn.softplus(self.gamma)
        return gamma * tf.math.sigmoid(alpha * x) + self.beta


class AffineScale(tf.keras.layers.Layer):
    """Trainable affine: softplus(gamma) * x + beta with a non-negative scale.

    Useful to rescale a bounded signal (e.g. a GRU output in [-1, 1]) so its
    magnitude can match unbounded encoder/decoder activations before concat.
    The scale is parameterized through softplus so it stays >= 0, and beta is a
    free per-channel bias. Applied on the last dimension.

    Args:
        gamma_init: Initial value for the (post-softplus) scale.
    """
    def __init__(self, gamma_init=1.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma_init = gamma_init
        self._fused = False

    def build(self, input_shape):
        dim = input_shape[-1]
        # inverse-softplus so that softplus(raw_init) == gamma_init
        raw_init = float(math.log(math.expm1(self.gamma_init)))
        self.gamma = self.add_weight(
            "gamma", shape=(dim,),
            initializer=tf.constant_initializer(raw_init),
            trainable=True)
        self.beta = self.add_weight(
            "beta", shape=(dim,),
            initializer="zeros",
            trainable=True)

    def fuse(self):
        """Precompute softplus(gamma) into a constant for inference."""
        self.gamma.assign(tf.nn.softplus(self.gamma))
        self._fused = True

    def call(self, x):
        gamma = self.gamma if self._fused else tf.nn.softplus(self.gamma)
        return gamma * x + self.beta


class DecomposablePReLU(tf.keras.layers.Layer):
    """PReLU decomposed for TFLite int16x8 compatibility.

    Training: relu(x) + alpha*(x - relu(x))
    After fuse():
      - If alpha is scalar → tf.nn.leaky_relu (1 FC, native LEAKY_RELU op)
      - If alpha is per-feature → keep decomposed form (2 FCs but fully quantized)
    """
    def __init__(self, shared_axes=None, **kwargs):
        super().__init__(**kwargs)
        self.shared_axes = shared_axes or [1, 2]
        self._fused = False
        self._fused_alpha = None

    def build(self, input_shape):
        param_shape = [1] * (len(input_shape) - 1)
        for i in range(1, len(input_shape)):
            if i not in self.shared_axes:
                param_shape[i - 1] = input_shape[i]
        self.alpha = self.add_weight(
            name='alpha',
            shape=param_shape,
            initializer='zeros',
            trainable=True)

    def fuse(self):
        """Freeze alpha; switch to leaky_relu if scalar."""
        alpha_np = self.alpha.numpy()
        if alpha_np.size == 1:
            self._fused_alpha = float(alpha_np.flat[0])
        else:
            self._fused_alpha = alpha_np
        self._fused = True

    def call(self, x):
        if self._fused:
            if isinstance(self._fused_alpha, float):
                return tf.nn.leaky_relu(x, alpha=self._fused_alpha)
            else:
                r = tf.nn.relu(x)
                return r + self._fused_alpha * (x - r)
        r = tf.nn.relu(x)
        return r + self.alpha * (x - r)


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
            self._sublayer = DecomposablePReLU(shared_axes=[1, 2], name='p_re_lu')
            self.fn = None
        elif act_name == "prelu_freq":
            self._sublayer = DecomposablePReLU(shared_axes=[1], name='p_re_lu')
            self.fn = None
        elif act_name == "dyt":
            self._sublayer = DyT(alpha_init=0.5, trainable_beta=False)
            self.fn = None
        elif act_name == "dys":
            self._sublayer = DyS(alpha_init=0.5, trainable_beta=False)
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

    def fuse(self):
        """Delegate fuse to sublayer (DyT, DyS)."""
        if self._sublayer is not None and hasattr(self._sublayer, 'fuse'):
            self._sublayer.fuse()