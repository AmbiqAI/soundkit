import tensorflow as tf

from soundkit.utils.tf_stft import gen_stft_win

class FrameClass:
    """
    Class to demonstrate the use of tf.signal.frame
    """
    def __init__(self, frame_length, frame_step, pad_end=False, pad_value=0):
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.pad_end = pad_end
        self.pad_value = pad_value
        self.state = None
        self.window_fn = gen_stft_win(
            win_size=frame_length,
            hop=frame_step
        )

    def apply_frames(self, audio_sn):
        """
        Apply frame transformation to the input tensor.
        """
        if self.state is None:
            shape = audio_sn.shape
            batch_size = shape[0]
            self.state = tf.zeros(
                (batch_size, self.frame_length - self.frame_step),
                dtype=tf.float32
            )

        audio_sn = tf.concat([self.state, audio_sn], axis=-1)

        self.state = audio_sn[:, -(self.frame_length - self.frame_step):]

        signals = tf.signal.frame(
            audio_sn,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            pad_end=self.pad_end,
            pad_value=self.pad_value
        )

        signals = signals * self.window_fn

        return signals

    def reconstruct(self, framed_signals):
        """
        Reconstruct the original signal from the framed signals.
        """
        framed_signals = framed_signals * self.window_fn
        reconstructed = tf.signal.overlap_and_add(framed_signals, frame_step=self.frame_step)
        overlap = self.frame_length - self.frame_step
        return reconstructed[:, overlap:-overlap]

    def reset_states(self):
        """
        Reset the state of the frame class.
        """
        self.state = None

if __name__ == "__main__":
    
    # Example input tensor with shape (batch_size, signal_length)
    x = tf.random.uniform((1, 10000), dtype=tf.float32)  # Example input tensor with shape (2, 50)
    x = tf.reshape(x, (1, -1))  # Example input tensor with shape (2, 50)
    import pdb; pdb.set_trace()  # Debugging breakpoint
    # x = tf.constant([[1,2,3,4,5,6, 7], [10,11,12,13,14,15, 16]], dtype=tf.float32)
    framing = FrameClass(frame_length=480, frame_step=160, pad_end=False, pad_value=0)
    y = framing.apply_frames(x)
    print(y)
    reconstructed = framing.reconstruct(y)

    x = tf.reshape(x, (-1,))
    reconstructed = tf.reshape(reconstructed, (-1,))

    x = x[:reconstructed.shape[0]]  # Ensure same length for comparison

    err = tf.reduce_min(tf.abs(x - reconstructed))

    print(f"Error: {err.numpy()}")

    print(reconstructed)