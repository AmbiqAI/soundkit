import tensorflow as tf

class ConfusionMatrixMetric(tf.keras.metrics.Metric):
    def __init__(self, num_classes: int, name="confusion_matrix", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.conf_matrix = self.add_weight(
            name="conf_matrix",    
            shape=(num_classes, num_classes),
            initializer="zeros",
            dtype=tf.int64,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        y_pred: shape (batch, ..., num_classes)
        y_true: shape (batch, ...) with integer class labels
        """
        # Convert logits/probs to predicted class indices
        pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int64)

        # Flatten both predictions and labels
        y_true_flat = tf.reshape(y_true, [-1])
        pred_flat = tf.reshape(pred_labels, [-1])

        # Compute confusion matrix for this batch
        batch_conf_matrix = tf.math.confusion_matrix(
            labels=y_true_flat,
            predictions=pred_flat,
            num_classes=self.num_classes,
            dtype=tf.int64
        )

        # Accumulate
        self.conf_matrix.assign_add(batch_conf_matrix)

    def result(self):
        """
        Returns the row-normalized confusion matrix as float64
        """
        conf_matrix_f64 = tf.cast(self.conf_matrix, tf.float64)
        row_sums = tf.reduce_sum(conf_matrix_f64, axis=1, keepdims=True)
        return tf.math.divide_no_nan(conf_matrix_f64, row_sums)

    def reset_states(self):
        self.conf_matrix.assign(tf.zeros_like(self.conf_matrix))
