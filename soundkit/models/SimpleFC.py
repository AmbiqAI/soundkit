from pydantic import BaseModel
import tensorflow as tf

class SimpleFCParams(BaseModel):
    units: int = 128
    dim_out: int = 257

class SimpleFC(tf.keras.Model):
    """Simple Fully Connected Neural Network with GRU layer.
    This model consists of two fully connected layers followed by a GRU layer
    and an output layer. It is designed for sequence prediction tasks.
    """
    def __init__(
            self,
            params: SimpleFCParams = SimpleFCParams(),
            **kwargs):
        """Initialize the SimpleFC model with given parameters."""
        super().__init__()
        self.params = params.units
        self.fc1 = tf.keras.layers.Dense(self.params.units, activation='relu')
        self.fc2 = tf.keras.layers.Dense(self.params.units, activation='relu')
        self.out = tf.keras.layers.Dense(self.params.dim_out, activation='sigmoid')

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)
