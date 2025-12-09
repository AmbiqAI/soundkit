import numpy as np
import soundfile as sf

def wind_noise(duration=5, fs=16000):
    decay = np.random.uniform(1, 2)
    n = int(duration * fs)
    freq_gust = np.random.uniform(0.1, 2)
    shift = np.random.uniform(1000, 100000)
    # Start with pink noise
    freqs = np.fft.rfftfreq(n, 1/fs)
    X = np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs))
    A = np.maximum(freqs, 1.0)**(-decay/2)  # ~pinkish shaping
    Y = X * A
    y = np.fft.irfft(Y, n=n)

    # Add slow amplitude modulation (simulate gusts)
    t = np.linspace(0, duration, n)
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * freq_gust * (t - shift))  # 0.2 Hz gusts
    y *= mod

    # Normalize
    y /= np.max(np.abs(y))
    return y.astype(np.float32)

samples = wind_noise(10)
sf.write("wind_noise.wav", samples, 16000)
