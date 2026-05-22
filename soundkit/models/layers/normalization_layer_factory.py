import tensorflow as tf

class NormalizationFactory(tf.keras.layers.Layer):
    def __init__(self, norm_type: str | None, **kwargs):
        # 1. ALWAYS call super().__init__() first
        # This initializes _trainable and other internal Keras mechanics
        super(NormalizationFactory, self).__init__(**kwargs)
        
        self.norm_type = norm_type.lower() if norm_type else None
        self.bn_layer = None

        if self.norm_type in ("batchnorm", "batchnorm_complex"):
            # 2. Use a unique name to avoid the "Variable:0" issue
            self.bn_layer = tf.keras.layers.BatchNormalization(
                momentum=0.995,
                center=False,
                scale=True,
                epsilon=1e-8,
                axis=-2,  # Normalize over the frequency dimension
                name=f"{self.name}_bn"
            )
        elif self.norm_type == "layernorm":
            self.bn_layer = tf.keras.layers.LayerNormalization(
                axis=[2, 3],
                name=f"{self.name}_ln"
            )
    def build(self, input_shape):
        # Create a variable to store the scale so it's saved in the checkpoint
        if self.norm_type == "batchnorm_complex":
            # Assuming freq dim is input_shape[2]
            freq_dim = input_shape[2]
            self.precomputed_scale = self.add_weight(
                name="precomputed_scale",
                shape=(1, 1, freq_dim, 1),
                initializer="ones",
                trainable=False
            )
        super(NormalizationFactory, self).build(input_shape)

    def update_inference_scale(self):
        """Call this before exporting to TFLite to bake the values."""
        sigma_sq = self.bn_layer.moving_variance
        gamma = self.bn_layer.gamma
        new_scale = gamma * tf.math.rsqrt(sigma_sq + self.bn_layer.epsilon)
        self.precomputed_scale.assign(tf.reshape(new_scale, (1, 1, -1, 1)))

    def call(self, x, training=False):
        """ call """
        if self.bn_layer is None:
            return x

        # --- Logic for BatchNorm (Pure Scaling, No Shifting) ---
        if self.norm_type == "batchnorm_complex":
            if training:
                # 1. Update statistics
                _ = self.bn_layer(x, training=training)

                # 2. Extract stats
                sigma_sq = self.bn_layer.moving_variance
                gamma = self.bn_layer.gamma

                # 3. Calculate pure multiplier: gamma / sqrt(sigma_sq + eps)
                scale = gamma * tf.math.rsqrt(sigma_sq + self.bn_layer.epsilon)
                # 4. FIXED ALIGNMENT:
                # Your x is [Batch, Time, Freq, Complex] -> [64, 800, 64, 2]
                # Your scale is [64] (from axis=-2)
                # We must reshape scale to [1, 1, 64, 1] so it hits the Freq dim
                scale = tf.reshape(scale, [1, 1, -1, 1])
                self.update_inference_scale()  # Update the variable for inference
                return x * self.precomputed_scale
            else:
                # INFERENCE PATH: Use the pre-baked scale
                # This avoids rsqrt and division ops entirely
                return x * self.precomputed_scale
            # return self.bn_layer(x, training=training)

        return self.bn_layer(x, training=training)
