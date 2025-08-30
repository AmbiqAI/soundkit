import tensorflow as tf

class PhaseLoss(tf.keras.losses.Loss):
    """
    Phase Loss for multi-class classification (softmax over last axis).

    Args:
        alpha: balancing factor (scalar or per-class list)
        gamma: focusing factor (> 0 reduces easy example weight)
    Input:
        y_true: (B, T) — integer class labels
        y_pred: (B, T, C) — softmax probabilities over classes
    """
    def __init__(self, eps=1e-8, name="phase_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.eps = eps

    def call(self, phase_true, phase_pred):
        """
        Compute the phase loss for multi-class classification with softmax output.

        phase_true: (B, T, F) — integer labels (0 or 1)
        phase_pred: (B, T, F) — softmax probs for class 0 and class 1
        """

        delta_theta = phase_pred - phase_true
        phase_loss = 1.0 - tf.math.cos(delta_theta)
        steps = tf.shape(phase_true)[0] * tf.shape(phase_true)[1]
        loss_tot = tf.reduce_sum(phase_loss, axis=[0, 1])  # Mean over batch and time

        return loss_tot / tf.cast(steps, tf.float32)
