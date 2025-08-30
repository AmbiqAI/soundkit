import tensorflow as tf

class NormalizationFactory(tf.keras.layers.Layer):
    def __init__(self, norm_type: str | None, **kwargs):
        # Pull out an outer name (for this wrapper layer) if provided
        outer_name = kwargs.pop("name", None)
        super().__init__(name=outer_name)   # <-- this was missing

        self.norm_type = norm_type.lower() if norm_type else None

        # Anything left in kwargs goes to the inner norm layer
        if self.norm_type == "layernorm":
            self.layer = tf.keras.layers.LayerNormalization(axis=[2, 3], **kwargs)
        elif self.norm_type == "batchnorm":
            self.layer = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9995, **kwargs)
        elif self.norm_type in (None, "none"):
            self.layer = None
        else:
            raise ValueError(f"Unsupported normalization layer: {norm_type}")

    def call(self, x, training=False):
        return x if self.layer is None else self.layer(x, training=training)
