""" LookaheadBuffer class for handling lookahead delay buffering."""
import tensorflow as tf

class LookaheadBuffer:
    """
    LookaheadBuffer handles lookahead delay buffering for streaming input features.

    This class is useful for models that require a small number of future frames
    (lookahead) while processing input in a streaming fashion.

    Attributes:
        num_lookahead (int): Number of lookahead frames to buffer.
        buffer (tf.Tensor): Internal buffer holding last `num_lookahead` frames.
    """

    def __init__(
            self,
            num_lookahead: int,
            feature_dim: int,
            batchsize: int,
            dtype: tf.DType = tf.complex64):
        """
        Initialize the LookaheadBuffer.

        Args:
            num_lookahead (int): Number of lookahead frames.
            feature_dim (int): Feature dimension (e.g., 257 for STFT).
            batchsize (int): Number of samples in the batch.
        """
        
        self.num_lookahead = num_lookahead
        self.buffer = tf.zeros([batchsize, num_lookahead, feature_dim], dtype=dtype)

    def apply(self, feat: tf.Tensor) -> tf.Tensor:
        """
        Apply lookahead delay to the input feature tensor.

        Args:
            feat (tf.Tensor): Input feature tensor of shape [B, T, F].

        Returns:
            tf.Tensor: Delayed tensor of shape [B, T, F], where the last
                       `num_lookahead` frames are dropped and replaced by past buffer.
        """
        if self.num_lookahead == 0:
            return feat

        delayed = tf.concat([self.buffer, feat], axis=1)
        feat_delayed = delayed[:, :-self.num_lookahead]
        self.buffer = delayed[:, -self.num_lookahead:]
        return feat_delayed

    def reset(self):
        """
        Reset the internal buffer to zeros.
        """
        self.buffer = tf.zeros_like(self.buffer)
