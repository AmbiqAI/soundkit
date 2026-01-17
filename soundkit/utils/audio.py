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
        target_length: int,
        pad_crop_mode: str="random") -> Tuple[np.ndarray, int, int]:
    """Pad audio to target length

    Args:
        audio (np.ndarray): audio data
        target_length (int): target length

    Returns:
        np.ndarray: padded audio data
    """
    if pad_crop_mode == "random":
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
            start = 0
            end = target_length
        
        
        else:
            start = 0
            end = target_length

        return audio, start, end

    elif pad_crop_mode == "tail":
        if len(audio) < target_length:
            zeros = np.zeros(target_length, dtype=audio.dtype)
            zeros[:len(audio)] = audio
            audio = zeros
            start = 0
            end = target_length

        elif len(audio) > target_length:
            audio = audio[:target_length]
            start = 0
            end = target_length
        
        else:
            start = 0
            end = target_length

        return audio, start, end
    else:
        raise ValueError(f"Unknown pad_crop_mode: {pad_crop_mode}. "
                         "Use 'random' or 'tail'.")


def get_labels(vad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dx = np.diff(vad)
    # Get indices where speech starts (0 → 1)
    onsets = np.where(dx > 0)[0] + 1

    # Get indices where speech ends (1 → 0)
    offsets = np.where(dx < 0)[0] + 1

    # Edge cases: if starts or ends with 1
    if vad[0] == 1:
        onsets = np.insert(onsets, 0, 0)
    if vad[-1] == 1:
        offsets = np.append(offsets, len(vad)-1)
    return offsets, onsets

def remove_short_segments(
        starts: np.ndarray,
        ends: np.ndarray,
        audio: np.ndarray,
        min_length: int = 160*10):
    '''Remove segments shorter than min_length
    Args:
        starts (np.ndarray): start indices of segments
        ends (np.ndarray): end indices of segments
        min_length (int): minimum length of segments
    Returns:
        Tuple[np.ndarray, np.ndarray]: filtered start and end indices
    '''
    valid_starts = []
    valid_ends = []
    vad = np.zeros_like(audio, dtype=np.float32)
    for s, e in zip(starts, ends):
        if e - s >= min_length:
            vad[s:e] = 1
            valid_starts.append(s)
            valid_ends.append(e)

    audio_update = audio * vad

    valid_starts = np.array(valid_starts, dtype=np.int32)
    valid_ends = np.array(valid_ends, dtype=np.int32)

    return valid_starts, valid_ends, audio_update

def pad_or_crop_with_labels(
        audio: np.ndarray,
        target_length: int,
        starts: np.ndarray,
        ends: np.ndarray,
        is_short_segments_remove: bool = True
        ) -> Tuple[np.ndarray, int, int]:
    """Pad audio to target length

    Args:
        audio (np.ndarray): audio data
        target_length (int): target length

    Returns:
        np.ndarray: padded audio data
    """
    vad = np.zeros_like(audio, dtype=np.int32)
    
    for s, e in zip(starts, ends):
        vad[s:e] = 1

    if len(audio) < target_length:

        sig = np.zeros(target_length, dtype=audio.dtype)
        start = np.random.randint(0, target_length - len(audio))
        end = start + len(audio)
        sig[start:end] = audio
        audio = sig

        sig = np.zeros(target_length, dtype=audio.dtype)
        sig[start:end] = vad
        vad = sig

    elif len(audio) > target_length:
        start = np.random.randint(0, len(audio) - target_length)
        end = start + target_length
        audio = audio[start:end]
        vad = vad[start:end]

    offsets, onsets = get_labels(vad)

    if is_short_segments_remove:
        onsets, offsets, audio = remove_short_segments(
            onsets, offsets, audio, min_length=160*10)

    return audio, onsets, offsets

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
        sample_rate: int,
        id: None | int = None) -> Tuple[np.ndarray, int, int]:
    """Randomly load audio from a list of audio files

    Args:
        audio_list (list): list of audio files
        target_length (int): target length

    Returns:
        np.ndarray: loaded audio data
    """
    
    if id is None:
        wav = np.random.choice(audio_list)
    else:
        wav = audio_list[id]
    audio = audio_read(wav, sample_rate = sample_rate)
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
    # if 0:
    if rir is not None:
        if 1:
            samples_75ms = sample_rate * 0.075 # 75ms

            idx_late_reverb = np.minimum(
                np.argmax(np.abs(rir)) + samples_75ms,
                rir.size-1).astype(np.int64)

            # rt60 = 1
            # dt = 1 / sample_rate
            # rt60_level = 10.0**(-60.0 / 5.0)
            # tau = -rt60 / np.log10(rt60_level)
            # n = np.arange(rir.size)

            # exponent = -(n - idx_late_reverb) * dt / tau
            # exponent = np.clip(exponent, -700, 0)  # adjust bounds as needed
            # decay = 10 ** exponent

            decay = np.ones(rir.size)
            decay[:idx_late_reverb] = 1
            decay[idx_late_reverb:] = 0

            rir_target = decay * rir
        else:
            idx = np.minimum(
                np.argmax(np.abs(rir)),
                rir.size-1).astype(np.int64)
            
            
            rir = rir[idx:]
            rir_target = rir*0
            rir_target[0]=rir[0]
        
        y = fftconvolve(signal_s, rir,'full')
        y = y[:len(signal_s)]
        target = fftconvolve(signal_s, rir_target, 'full')
        target = target[:len(signal_s)]

        # import matplotlib.pyplot as plt
        # plt.subplot(4,1,1)
        # plt.plot(signal_s)
        # plt.xlim([0, 16000 * 5])
        # plt.subplot(4,1,2)
        # plt.plot(y)
        # plt.xlim([0, 16000 * 5])
        # plt.subplot(4,1,3)
        # plt.plot(target)
        # plt.xlim([0, 16000 * 5])
        # plt.subplot(4,1,4)
        # plt.plot(decay)
        # plt.xlim([0, 16000 * 5])
        # plt.show()
    else:
        y = signal_s.copy()
        target = signal_s.copy()

    # Compute clean and noise powers
    clean_power = np.mean(target**2)

    if clean_power == 0:
        pass
    else:
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
    y *= gain
    return signal_sn, target, start_index, end_index


def synthesize_audio_with_labels(
        clean: np.ndarray,
        noise: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
        rir: np.ndarray,
        snr_db: float,
        min_amp: float = 0.01,
        max_amp: float = 0.95,
        target_length: int = 160*500,
        sample_rate:int = 16000,
        is_short_segments_remove=True) -> Tuple[np.ndarray, np.ndarray, int, int]:
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
    signal_s, onsets, offsets = pad_or_crop_with_labels(
        clean, target_length,
        starts, ends,
        is_short_segments_remove)
    noise = repeat_or_crop(noise, target_length)
    # if 0:

    if rir is not None:
        samples_50ms = int(sample_rate * 0.05) # 50ms
        y = fftconvolve(signal_s, rir,'full')
        y = y[:len(signal_s)]
        delay = np.argmax(np.abs(rir)) + samples_50ms
        onsets[0] = np.minimum(onsets[0]+delay, len(signal_s)-1)
        offsets[0] = np.minimum(offsets[0]+delay, len(signal_s)-1)

    else:
        y = signal_s.copy()

    # Compute clean and noise powers

    clean_power=np.mean(y**2)
 
    # steps=0
    # for s,e in zip(onsets, offsets):
    #     clean_power += np.sum(target[s:e]**2)
    #     steps+=e-s+1
    # if steps > 0:
    #     clean_power /= steps
    if clean_power == 0:
        pass
    else:
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
    y *= gain
    signal_sn *= gain

    return signal_sn, y, onsets, offsets

def synthesize_audio_with_labels_vad(
        clean: np.ndarray,
        noise: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
        rir: np.ndarray,
        snr_db: float,
        min_amp: float = 0.01,
        max_amp: float = 0.95,
        target_length: int = 160*500,
        sample_rate:int = 16000,
        is_short_segments_remove=True) -> Tuple[np.ndarray, np.ndarray, int, int]:
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
    signal_s, onsets, offsets = pad_or_crop_with_labels(
        clean, target_length,
        starts, ends,
        is_short_segments_remove)
    noise = repeat_or_crop(noise, target_length)
    # if 0:

    if rir is not None:
        idx = np.argmax(np.abs(rir))
        idx = np.maximum(idx-20, 0)
        h_aligned = rir[idx:]
        y_full = fftconvolve(signal_s, h_aligned, mode="full")
        y = y_full[: len(signal_s)]  # keep first T samples; discard late tail
        target = y.copy()

        if 0:
            vad = np.zeros_like(signal_s, dtype=np.float32)
            for ss, ee in zip(onsets, offsets):
                vad[ss:ee] = 0.9
            import matplotlib.pyplot as plt
            plt.subplot(5,1,1)
            plt.plot(signal_s)
            plt.plot(vad)
            # plt.xlim([0, 16000 * 5])
            plt.subplot(5,1,2)
            plt.plot(y)
            plt.plot(vad)

            plt.subplot(5,1,3)
            plt.plot(rir)

            plt.subplot(5,1,4)
            plt.plot(h_aligned)
            # plt.xlim([0, 16000 * 5])
        
            plt.show()
        
    else:
        y = signal_s.copy()
        target = signal_s.copy()

    # Compute clean and noise powers

    clean_power=0
    steps=0
    for s,e in zip(onsets, offsets):
        clean_power += np.sum(target[s:e]**2)
        steps+=e-s+1
    if steps > 0:
        clean_power /= steps
    if clean_power == 0:
        pass
    else:
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
    y *= gain
    return signal_sn, target, onsets, offsets