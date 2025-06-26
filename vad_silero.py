import torch
from soundkit.utils.audio import audio_read
import numpy as np

# Force reload to get the latest version (optional, but useful)
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=True
)

# If utils is a tuple
get_speech_timestamps = utils[0]

# Read audio and reshape if necessary
import os, re, json
lst=[]
i = 0
for root, dirs, files in os.walk('wavs/LibriSpeech/train-other-500'):
    for fname in files:

        if re.search(r'(\.wav$|\.flac$)', fname):
            print(f"\r{i}", end='')
            i += 1
            path = os.path.join(root, fname)

            lst.append(path)

# Write to CSV
import csv, json
from tqdm import tqdm  # for progress bar
with open("output.csv", mode="w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "label"])
    for path in tqdm(lst, desc="Processing audio files"):

        wav = audio_read(path, sample_rate=16000)
        timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)

        if len(timestamps) != 0:
            starts = np.array([item['start'] for item in timestamps])
            ends = np.array([item['end'] for item in timestamps])
            vad = np.zeros_like(wav, dtype=np.float32)
            for s,e in zip(starts, ends):
                vad[s:e] = 1.0

            import matplotlib.pyplot as plt
            plt.plot(wav)
            plt.plot(vad)
            plt.show()
            label_str = json.dumps(timestamps)  # properly escaped JSON
            writer.writerow([path, label_str])
        else:
            print(path)

# import numpy as np
# # Decode to separate numpy arrays
# starts = np.array([item['start'] for item in timestamps])
# ends = np.array([item['end'] for item in timestamps])

# vad = np.zeros_like(wav, dtype=np.float32)
# for s,e in zip(starts, ends):
#     vad[s:e] = 1.0
# import pdb;
# import matplotlib.pyplot as plt
# plt.plot(wav)
# plt.plot(vad)
# plt.show()