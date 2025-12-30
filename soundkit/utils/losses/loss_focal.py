"""Focal Loss Implementation"""
import tensorflow as tf

class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for multi-class classification (softmax over last axis).

    Args:
        alpha: balancing factor (scalar or per-class list)
        gamma: focusing factor (> 0 reduces easy example weight)
    Input:
        y_true: (B, T) — integer class labels
        y_pred: (B, T, C) — softmax probabilities over classes
    """
    def __init__(self, alpha=0.75, gamma=2.0, name="focal_loss", **kwargs):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        """
        Compute the focal loss for binary classification with softmax output.

        y_true: (B, T) — integer labels (0 or 1)
        y_pred: (B, T, 2) — softmax probs for class 0 and class 1
        """
        y_true = tf.cast(y_true, tf.int32)                 # (B, T)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Gather predicted prob of the true class
        indices = tf.expand_dims(y_true, axis=-1)          # (B, T, 1)
        p_t = tf.gather(y_pred, indices=indices, axis=-1, batch_dims=2)  # (B, T, 1)
        p_t = tf.squeeze(p_t, axis=-1)                     # => (B, T)

        # Class-specific alpha_t
        alpha_t = tf.where(
            tf.equal(y_true, 1),
            self.alpha,
            1.0 - self.alpha)  # (B, T)

        # Focal loss
        focal_weight = tf.pow(1.0 - p_t, self.gamma)       # (B, T)
        loss = -alpha_t * focal_weight * tf.math.log(p_t)  # (B, T)

        return tf.reduce_mean(loss)
