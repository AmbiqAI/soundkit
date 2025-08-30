""" stft and istft functions for tensorflow
    Those function support batch processing
"""
import tensorflow as tf
import math as m
pi = tf.constant(m.pi)

def gen_stft_win(
        win_size: int     = 240,
        hop : int         = 80):
    """
    STFT window generator
    """

    # if win_size % hop != 0:
    #     raise ValueError("Window size must be a multiple of hop size")
    if win_size == hop:
        win =  tf.ones(win_size, dtype=tf.float32)
    else:
        win_size = tf.cast(win_size, tf.float32)
        hop_div_fr = tf.cast(hop,tf.float32) / win_size
        indices = tf.range(win_size, dtype=tf.float32)

        win_square = hop_div_fr * (1.0 - tf.math.cos(2.0 * pi / win_size * indices ))
        win = tf.sqrt(win_square)

    return win

def window_fn(
        frame_length: int,
        frame_step: int,
        dtype=tf.float32):
    """ Create a window function"""

    win = gen_stft_win(
            win_size        = frame_length,
            hop             = frame_step)

    return win

@tf.function
def tf_stft(
        signals: tf.float32,
        frame_length: int,
        frame_step: int,
        fft_length: int,
        states=None,):
    
    """ Calculate the STFT of a batch of signals"""
    def my_win(frame_length, dtype=tf.float32):
        return window_fn(frame_length, tf.constant(frame_step),dtype=dtype)
    shape = tf.shape(signals)

    if states is None:
        states = tf.zeros([shape[0], frame_length-frame_step])
    signals = tf.concat([states, signals], axis=-1)
    signals_stft = tf.signal.stft(
        signals,
        frame_length=frame_length,  # Length of each frame in samples
        frame_step=frame_step,   # Number of samples to shift between frames
        window_fn=my_win,
        fft_length=fft_length     # Length of the FFT
    )
    return signals_stft

@tf.function
def tf_istft(
        signals_stft: tf.float32,
        frame_length: int,
        frame_step: int,
        fft_length: int):
    """ Calculate the inverse STFT of a batch of signals"""
    def my_win(frame_length, dtype=tf.float32):
        return window_fn(frame_length, tf.constant(frame_step),dtype=dtype)
    
    signals_recons = tf.signal.inverse_stft(
        signals_stft,
        frame_length = frame_length,
        frame_step = frame_step,
        fft_length=fft_length,
        window_fn=my_win,
        name=None
    )
    overlap=frame_length-frame_step
    return signals_recons[:,overlap:-overlap]

class StreamingSTFT(tf.Module):
    def __init__(
            self,
            frame_length=480,
            frame_step=160,
            fft_length=512,
            window_fn=window_fn):
        super().__init__()
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.buffer_size = frame_length - frame_step
        self.window = window_fn(frame_length, frame_step)
        self.reset()

    def reset(self):
        self.buffer = tf.zeros([self.buffer_size], dtype=tf.float32)

    def process_frame(self, audio_chunk):
        """
        Input: audio_chunk [frame_step]
        Output: stft_frame [frame_length//2 + 1] (complex)
        """

        full_frame = tf.concat([self.buffer, audio_chunk], axis=0)

        self.buffer = full_frame[self.frame_step:]
        windowed = full_frame * self.window
        return tf.signal.rfft(windowed, [self.fft_length])


class StreamingISTFT(tf.Module):
    def __init__(
            self,
            frame_length=480,
            frame_step=160,
            fft_length=512,
            window_fn=window_fn):
        super().__init__()
        self.frame_length = frame_length
        self.fft_length = fft_length
        self.frame_step = frame_step
        self.window = window_fn(frame_length, frame_step)
        self.reset()

    def reset(self):
        self.overlap_buffer = tf.zeros([self.frame_length - self.frame_step], dtype=tf.float32)

    def process_frame(self, stft_frame):
        """
        Input: stft_frame [frame_length//2 + 1] (complex)
        Output: waveform_chunk [frame_step]
        """
        time_frame = tf.signal.irfft(stft_frame, [self.fft_length])
        
        windowed = time_frame * self.window

        # Apply overlap-add
        full = tf.concat([self.overlap_buffer, tf.zeros([self.frame_step], dtype=tf.float32)], axis=0)
        full += windowed

        # Output the first hop-size samples
        out = full[:self.frame_step]

        # Update overlap buffer
        self.overlap_buffer = full[self.frame_step:self.frame_length]
        return out
