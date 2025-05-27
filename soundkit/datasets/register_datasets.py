# soundkit/plugins/register_datasets.py

import random
import os
import re
from pathlib import Path
from .dataset_registry import DatasetRegistry


def get_wavefiles(path_folder):
    lst = []
    for root, _, files in os.walk(f'{path_folder}'):
        for file in files:
            if re.search(r'(wav$|flac$)', file):
                lst += [os.path.join(root, file.strip())]
    return lst

# === Train/Val Corpora ===

@DatasetRegistry.register("train-clean-100")
def load_train_clean_100(corpus):
    return get_wavefiles("wavs/LibriSpeech/train-clean-100")


@DatasetRegistry.register("train-clean-360")
def load_train_clean_360(corpus):
    return get_wavefiles("wavs/LibriSpeech/train-clean-360")


@DatasetRegistry.register("dev-clean")
def load_dev_clean(corpus):
    return get_wavefiles("wavs/LibriSpeech/dev-clean")


@DatasetRegistry.register("test-clean")
def load_test_clean(corpus):
    return get_wavefiles("wavs/LibriSpeech/test-clean")

@DatasetRegistry.register("thchs30")
def load_thchs30(corpus):
    train_list = get_wavefiles("wavs/data_thchs30/train")
    dev_list = get_wavefiles("wavs/data_thchs30/dev")
    return {"train": train_list, "val": dev_list}


# === Noise Corpora ===

@DatasetRegistry.register("wham_noise")
def load_wham_noise(corpus):
    files = {}
    files['train'] = get_wavefiles('wavs/noise/wham_noise/tr')
    files['val'] = get_wavefiles('wavs/noise/wham_noise/cv')
    return files


@DatasetRegistry.register("FSD50K")
def load_fsd50k(corpus):
    with open('wavs/noise/FSD50K/non_speech.csv', 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    random.shuffle(lines)
    split = len(lines) // 5
    return {"train": lines[split:], "val": lines[:split]}


@DatasetRegistry.register("ESC-50-master")
def load_esc50(corpus):
    with open('wavs/noise/ESC-50-master/non_speech.csv', 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    random.shuffle(lines)
    split = len(lines) // 5
    return {"train": lines[split:], "val": lines[:split]}


@DatasetRegistry.register("musan")
def load_musan(corpus):
    music = get_wavefiles('wavs/noise/musan/music')
    noise = get_wavefiles('wavs/noise/musan/noise')
    lines = music + noise
    random.shuffle(lines)
    split = len(lines) // 5
    return {"train": lines[split:], "val": lines[:split]}


# === Reverb Corpus ===

@DatasetRegistry.register("RIRS_NOISES")
@DatasetRegistry.register("rirs_noises")
def load_rirs_noises(corpus):
    all_files = get_wavefiles('wavs/noise/RIRS_NOISES')
    random.shuffle(all_files)
    split = len(all_files) // 5
    return {"train": all_files[split:], "val": all_files[:split]}
