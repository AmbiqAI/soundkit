import tensorflow as tf

class NormalizationFactory:
    def __init__(self, norm_type: str | None, **kwargs):
        self.norm_type = norm_type.lower() if norm_type else None

        if self.norm_type == "layernorm":
            self.layer = tf.keras.layers.LayerNormalization(
                axis=[2, 3], **kwargs)
        elif self.norm_type == "batchnorm":
            self.layer = tf.keras.layers.BatchNormalization(
                axis=[2, 3], momentum=0.9995, **kwargs)
        elif self.norm_type in (None, "none"):
            self.layer = None
        else:
            raise ValueError(f"Unsupported normalization layer: {norm_type}")

    def __call__(self, x, training=False):
        if self.layer is None:
            return x
        return self.layer(x, training=training)
