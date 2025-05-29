'''Feature extraction utilities for audio signals.
This module provides functions to read audio files, pad or crop audio signals,
synthesize noisy audio, and handle various audio processing tasks.'''
from typing import Tuple
import soundfile as sf
import librosa
import numpy as np
from scipy.signal import fftconvolve

def audio_read(
        fname:str,
        sample_rate: int = 16000,) -> np.ndarray:
    """Read audio file

    Args:
        fname (str): audio file name
        dtype (str): data type

    Returns:
        np.ndarray: audio data
    """
    try:
        data, sample_rate_orig = sf.read(fname)
        if data.ndim > 1:
            data=data[:,0]

        if sample_rate < sample_rate_orig:
            data = librosa.resample(
                        data,
                        orig_sr=sample_rate_orig,
                        target_sr=sample_rate)

        return data

    except Exception as e:
        raise RuntimeError(f"Failed to read {fname}: {e}")

def pad_or_crop(
        audio: np.ndarray,
        target_length: int) -> Tuple[np.ndarray, int, int]:
    """Pad audio to target length

    Args:
        audio (np.ndarray): audio data
        target_length (int): target length

    Returns:
        np.ndarray: padded audio data
    """
    if len(audio) < target_length:

        zeros = np.zeros(target_length, dtype=audio.dtype)
        start = np.random.randint(0, target_length - len(audio))
        end = start + len(audio)
        zeros[start:end] = audio
        audio = zeros

    elif len(audio) > target_length:
        start = np.random.randint(0, len(audio) - target_length)
        end = start + target_length
        audio = audio[start:end]
    else:
        start = 0
        end = target_length

    return audio, start, end

def repeat_or_crop(x: np.ndarray, target_length: int) -> np.ndarray:
    """
    Repeat or crop a 1D signal to match the target length.

    Args:
        x (np.ndarray): Input 1D array (e.g., noise)
        target_length (int): Desired output length

    Returns:
        np.ndarray: Output array of shape (target_length,)
    """
    if len(x) == 0:
        raise ValueError("Input signal is empty.")

    if len(x) < target_length:
        # Repeat as many times as needed and crop the excess
        n_repeats = (target_length + len(x) - 1) // len(x)
        x = np.tile(x, n_repeats)[:target_length]
    elif len(x) > target_length:
        # Random crop
        start = np.random.randint(0, len(x) - target_length)
        x = x[start:start + target_length]
    # else, already the right length

    return x


def random_load_audio_from_list(
        audio_list: list,
        sample_rate: int) -> Tuple[np.ndarray, int, int]:
    """Randomly load audio from a list of audio files

    Args:
        audio_list (list): list of audio files
        target_length (int): target length

    Returns:
        np.ndarray: loaded audio data
    """
    idx = np.random.randint(0, len(audio_list))
    audio = audio_read(audio_list[idx], sample_rate = sample_rate)
    return audio

def synthesize_audio(
        clean: np.ndarray,
        noise: np.ndarray,
        rir: np.ndarray,
        snr_db: float,
        min_amp: float = 0.01,
        max_amp: float = 0.95,
        target_length: int = 160*500,
        sample_rate:int = 16000) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Synthesize noisy audio from clean and noise with optional reverberation.

    Args:
        clean (np.ndarray): Clean speech audio
        noise (np.ndarray): Noise audio
        rir (np.ndarray): Room impulse response (can be None or empty)
        snr_db (float): Target signal-to-noise ratio in dB
        min_amp (float): Minimum peak amplitude
        max_amp (float): Maximum peak amplitude
        target_length (int): Target audio length
        sample_rate (int): Sample rate of the audio

    Returns:
        Tuple[np.ndarray, np.ndarray, int, int]: 
            (noisy_audio, clean_audio, start_index, end_index)
    """
    signal_s, start_index, end_index = pad_or_crop(clean, target_length)
    noise = repeat_or_crop(noise, target_length)
    if rir is not None:
        samples_5ms = sample_rate * 0.005 # 5ms

        idx_late_reverb = np.minimum(
            np.argmax(np.abs(rir)) + samples_5ms,
            rir.size-1).astype(np.int64)

        rt60 = 1
        dt = 1 / sample_rate
        rt60_level = 10.0**(-60.0 / 5.0)
        tau = -rt60 / np.log10(rt60_level)
        n = np.arange(rir.size)

        exponent = -(n - idx_late_reverb) * dt / tau
        exponent = np.clip(exponent, -700, 0)  # adjust bounds as needed
        decay = 10 ** exponent

        decay[:idx_late_reverb] = 1

        rir_target = decay * rir
        y = fftconvolve(signal_s, rir,'same')
        target = fftconvolve(signal_s, rir_target, 'same')
        # import matplotlib.pyplot as plt
        # plt.subplot(4,1,1)
        # plt.plot(rir)
        # plt.subplot(4,1,2)
        # plt.plot(rir_target)
        # plt.subplot(4,1,3)
        # plt.plot(rir - rir_target)
        # plt.subplot(4,1,4)
        # plt.plot(decay)
        
        # plt.show()
    
    else:
        y = signal_s.copy()
        target = signal_s.copy()

    # Compute clean and noise powers
    clean_power = np.mean(target**2) + 1e-9
    noise_power = np.mean(noise**2) + 1e-9

    # Scale noise to match desired SNR
    desired_noise_power = clean_power / (10**(snr_db / 10))
    scale = np.sqrt(desired_noise_power / noise_power)
    noise = noise * scale

    # Mix
    signal_sn = y + noise

    # Normalize to avoid clipping
    peak = max(np.max(np.abs(signal_sn)), 1e-8)
    gain = 1 / peak * np.random.uniform(min_amp, max_amp)
    target *= gain
    signal_sn *= gain

    return signal_sn, target, start_index, end_index