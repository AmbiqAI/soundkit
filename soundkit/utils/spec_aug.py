import tensorflow as tf
import matplotlib.pyplot as plt

class SpecAug:
    def __init__(
            self,
            freq_mask_width=15,
            num_freq_mask=2,
            time_mask_width=0.0,
            num_time_mask=0,
            prob=0.5):
        """
        SpecAugment with frequency and time masking.
        Args:
            freq_mask_width: max width of frequency mask (int)
            num_freq_mask: number of freq masks per sample
            time_mask_width: max width of time mask (as percent of total time, float in (0,1])
            num_time_mask: number of time masks per sample
        """
        self.prob = prob
        self.freq_mask_width = freq_mask_width
        self.num_freq_mask = num_freq_mask
        self.time_mask_width = time_mask_width
        self.num_time_mask = num_time_mask

    def _apply_freq_mask(self, x, y):
        v = tf.random.uniform([])
        if v > self.prob:
            return x, y
            
        batch_size = tf.shape(x)[0]
        time = tf.shape(x)[1]
        freq = tf.shape(x)[2]

        width = tf.random.uniform((batch_size,), minval=1, maxval=self.freq_mask_width + 1, dtype=tf.int32)
        max_start = freq - width
        start = tf.random.uniform((batch_size,), minval=0, maxval=freq // 5, dtype=tf.int32)
        start = tf.minimum(start, max_start)

        freq_range = tf.expand_dims(tf.range(freq), 0)          # (1, freq)
        freq_range = tf.tile(freq_range, [batch_size, 1])       # (batch, freq)

        mask = tf.logical_or(freq_range < tf.expand_dims(start, 1),
                             freq_range >= tf.expand_dims(start + width, 1))  # (batch, freq)
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, 1)                          # (batch, 1, freq)
        mask = tf.tile(mask, [1, time, 1])                      # (batch, time, freq)

        noise = tf.ones_like(x) * 10
        gain = tf.random.uniform((batch_size, 1, 1), minval=0.1, maxval=0.5)
        if x.dtype == tf.complex64:
            mask = tf.complex(mask, tf.zeros_like(mask))
            gain = tf.cast(gain, tf.complex64)
        x = mask * x + (1 - mask) * x * gain
        y = mask * y + (1 - mask) * y * gain


        return x, y

    def _apply_time_mask(self, x, y):
        batch_size = tf.shape(x)[0]
        time = tf.shape(x)[1]
        freq = tf.shape(x)[2]

        max_width = tf.cast(tf.math.maximum(1.0, self.time_mask_width * tf.cast(time, tf.float32)), tf.int32)
        width = tf.random.uniform((batch_size,), minval=1, maxval=max_width + 1, dtype=tf.int32)
        max_start = time - width
        start = tf.random.uniform((batch_size,), minval=0, maxval=time, dtype=tf.int32)
        start = tf.minimum(start, max_start)

        time_range = tf.expand_dims(tf.range(time), 0)          # (1, time)
        time_range = tf.tile(time_range, [batch_size, 1])       # (batch, time)

        mask = tf.logical_or(time_range < tf.expand_dims(start, 1),
                             time_range >= tf.expand_dims(start + width, 1))  # (batch, time)
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, 2)                          # (batch, time, 1)
        mask = tf.tile(mask, [1, 1, freq])                      # (batch, time, freq)

        gain = tf.random.uniform((batch_size, 1, 1), minval=0.1, maxval=0.5)
        if x.dtype == tf.complex64:
            mask = tf.complex(mask, tf.zeros_like(mask))
            gain = tf.cast(gain, tf.complex64)
        x = mask * x + (1 - mask) * x * gain
        y = mask * y + (1 - mask) * y * gain

        return x, y

    def __call__(self, x, y):
        for _ in range(self.num_freq_mask):
            x, y = self._apply_freq_mask(x, y)
        for _ in range(self.num_time_mask):
            x, y = self._apply_time_mask(x, y)
        return x, y