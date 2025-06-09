# soundkit/plugins/register_datasets.py

import random
import os
import re
from pathlib import Path
import csv
import json

def load_wav_label_csv(lst: str, filter=None) -> list:
    """
    Load VAD data from a CSV file.
    The CSV file should have two columns: 'filename' and 'label'.
    The 'label' column contains JSON strings representing lists of dictionaries.
    """
    data = []
    with open(lst, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            label = json.loads(row["label"])  # Parse JSON string back to list
            data.append((fname, label))

    if filter is not None:
        data = [item for item in data if re.search(filter, item[0])]

    return data

def get_wavefiles(path_folder):
    lst = []
    for root, _, files in os.walk(f'{path_folder}'):
        for file in files:
            if re.search(r'(wav$|flac$)', file):
                lst += [os.path.join(root, file.strip())]
    return lst

# === Train/Val Corpora ===
def load_train_clean_100(corpus):
    return get_wavefiles("wavs/LibriSpeech/train-clean-100")


def load_train_clean_360(corpus):
    return get_wavefiles("wavs/LibriSpeech/train-clean-360")


def load_dev_clean(corpus):
    return get_wavefiles("wavs/LibriSpeech/dev-clean")


def load_test_clean(corpus):
    return get_wavefiles("wavs/LibriSpeech/test-clean")

def load_thchs30(corpus):
    train_list = get_wavefiles("wavs/data_thchs30/train")
    dev_list = get_wavefiles("wavs/data_thchs30/dev")
    return {"train": train_list, "val": dev_list}
# for kws
def load_train_galaxy(corpus):
    return load_wav_label_csv('data/galaxy_train.csv')

def load_val_galaxy(corpus):
    return load_wav_label_csv('data/galaxy_val.csv')

# for vad
def load_vad_train_clean_100(corpus, filter='train-clean-100'):
    return load_wav_label_csv('data/vad_train_labels.csv', filter)

def load_vad_train_clean_360(corpus, filter='train-clean-360'):
    return load_wav_label_csv('data/vad_train_labels.csv', filter)

def load_vad_dev_clean(corpus, filter="dev-clean"):
    return load_wav_label_csv('data/vad_val_labels.csv', filter)

def load_vad_thchs30(corpus):

    train_list = load_wav_label_csv('data/vad_train_labels.csv', "wavs/data_thchs30/train")
    dev_list = load_wav_label_csv('data/vad_val_labels.csv', "wavs/data_thchs30/dev")

    return {"train": train_list, "val": dev_list}

# === Noise Corpora ===

def load_wham_noise(corpus):
    files = {}
    files['train'] = get_wavefiles('wavs/noise/wham_noise/tr')
    files['val'] = get_wavefiles('wavs/noise/wham_noise/cv')
    return files


def load_fsd50k(corpus):
    with open('data/FSD50K/non_speech.csv', 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    random.shuffle(lines)
    split = len(lines) // 5
    return {"train": lines[split:], "val": lines[:split]}


def load_esc50(corpus):
    with open('data/ESC-50-master/non_speech.csv', 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    random.shuffle(lines)
    split = len(lines) // 5
    return {"train": lines[split:], "val": lines[:split]}


def load_musan(corpus):
    music = get_wavefiles('wavs/noise/musan/music')
    noise = get_wavefiles('wavs/noise/musan/noise')
    lines = music + noise
    random.shuffle(lines)
    split = len(lines) // 5
    return {"train": lines[split:], "val": lines[:split]}


# === Reverb Corpus ===

def load_rirs_noises(corpus):
    all_files = get_wavefiles('wavs/noise/RIRS_NOISES')
    random.shuffle(all_files)
    split = len(all_files) // 5
    return {"train": all_files[split:], "val": all_files[:split]}
