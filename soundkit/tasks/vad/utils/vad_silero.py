import torch
from soundkit.utils.audio import audio_read
import numpy as np

# Force reload to get the latest version (optional, but useful)
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=True
)

def get_vad(wav, sampling_rate=16000):
    get_speech_timestamps = utils[0]
    timestamps = get_speech_timestamps(
        wav, model, sampling_rate=sampling_rate)
    vad = np.zeros_like(wav, dtype=np.float32)
    if len(timestamps) != 0:
        starts = np.array([item['start'] for item in timestamps])
        ends = np.array([item['end'] for item in timestamps])

        for s,e in zip(starts, ends):
            vad[s:e] = 1.0
    return vad

def calculate_vad_accuracy(vad, vad_gt):
    # assert len(vad) == len(vad_gt), "VAD and ground truth must have the same length"
    error = vad != vad_gt

    error = error.astype(np.int64)
    mask = vad_gt == 0
    fa = error[mask].sum()  # False Alarms
    fa_total = mask.sum()
    fa = fa / fa_total

    mask = vad_gt == 1
    fr = error[mask].sum()  # False Rejections
    fr_total = mask.sum()
    fr = fr / fr_total

    return fa, fr, fa_total, fr_total



if __name__ == "__main__":
    sig = audio_read('crest_vad/MixedNoise_In.wav', sample_rate=16000)
    import soundfile as sf
    vad,fs = sf.read('soundkit/tasks/vad/test_results/model/MixedNoise_In_vad.wav')

    vad_gt = get_vad(sig, sampling_rate=16000)
    vad_gt = vad_gt[:len(vad)]  # Ensure both arrays are the same length

    # vad    = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0], dtype=np.float32)
    # vad_gt = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 0], dtype=np.float32)

    vad = (vad > 0.1).astype(np.int16)
    vad_gt = vad_gt.astype(np.int16)
    fa, fr, fa_total, fr_total = calculate_vad_accuracy(vad, vad_gt)

    print(f"False Alarms: {fa:.4f} (total: {fa_total})")
    print(f"False Rejections: {fr:.4f} (total: {fr_total})")
