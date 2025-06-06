import numpy as np
from soundkit.utils.tf_stft import gen_stft_win

class StreamingSTFT:
    """
    Stateful streaming STFT processor using overlap and windowing.
    Feeds audio chunks and returns new STFT frames when available.
    """
    def __init__(self, frame_len=480, hop_len=160, fft_len=512):
        """
        Args:
            frame_len (int): Length of each STFT frame in samples.
            hop_len (int): Hop size (stride) in samples.
            fft_len (int): FFT size. Defaults to frame_len if None.
            window (str): Window function name (e.g., 'hann').
        """
        self.frame_len = frame_len
        self.hop_len = hop_len
        self.fft_len = fft_len or frame_len
        self.window = gen_stft_win(frame_len, hop_len).numpy()
        self.reset()

    def process(self, chunk):
        """
        Process a chunk of audio samples and return a list of STFT frames.

        Args:
            chunk (np.ndarray): 1D float32 array of new audio samples.

        Returns:
            list of np.ndarray: Complex STFT frames (rfft).
        """
        self.buffer = np.concatenate([self.buffer, chunk])
        frame = self.buffer[:self.frame_len] * self.window
        stft_frame = np.fft.rfft(frame, n=self.fft_len)
        self.buffer = self.buffer[self.hop_len:]  # slide window

        return stft_frame

    def reset(self):
        """Reset the internal buffer."""
        self.buffer = np.zeros(self.frame_len - self.hop_len, dtype=np.float32)


class StreamingISTFT:
    """
    Stateful streaming inverse STFT with overlap-add reconstruction.
    Keeps a rolling buffer and outputs hop_len new samples per frame.
    """
    def __init__(self, frame_len=480, hop_len=160, fft_len=512):
        """
        Args:
            frame_len (int): STFT frame length in samples.
            hop_len (int): Hop size in samples.
            fft_len (int): FFT size. Defaults to frame_len if None.
            window (str): Window function name (e.g., 'hann').
        """
        self.frame_len = frame_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        self.window = gen_stft_win(frame_len, hop_len).numpy()
        self.reset()

    def process(self, stft_frame):
        """
        Process a single STFT frame and return the next hop_len audio samples.

        Args:
            stft_frame (np.ndarray): Complex 1D STFT frame (rfft output).

        Returns:
            np.ndarray: 1D float32 array of hop_len samples.
        """
        time_frame = np.fft.irfft(stft_frame, n=self.fft_len)[:self.frame_len]
        time_frame *= self.window
        self.buffer += time_frame

        output = self.buffer[:self.hop_len].copy()
        self.buffer[:-self.hop_len] = self.buffer[self.hop_len:]
        self.buffer[-self.hop_len:] = 0.0

        return output.astype(np.float32)

    def reset(self):
        """Reset the overlap-add buffer."""
        self.buffer = np.zeros(self.frame_len, dtype=np.float32)


if __name__ == "__main__":
    # Example usage
    stft = StreamingSTFT(frame_len=480, hop_len=160, fft_len=512)
    istft = StreamingISTFT(frame_len=480, hop_len=160, fft_len=512)

    # Simulate processing audio chunks
    audio_chunks = [np.random.randn(160) for _ in range(1000)]  # 10 chunks of 160 samples
    out = []
    for chunk in audio_chunks:
        stft_frame = stft.process(chunk)
        output_samples = istft.process(stft_frame)
        out += [output_samples]

    audio = np.concatenate(out)[320:-320]  # Remove initial buffer artifacts
    out = np.concatenate(out)[320:-320]
    e = np.abs(audio-out).max()
    print(f"Max error between original and reconstructed audio: {e:.6f}")
