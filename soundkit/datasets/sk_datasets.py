# soundkit/plugins/register_datasets.py
import random
import os
import re
import csv
import json

corpus2path_map= {
    "train-clean-100": "wavs/LibriSpeech/train-clean-100",
    "train-clean-360": "wavs/LibriSpeech/train-clean-360",
    "dev-clean": "wavs/LibriSpeech/dev-clean",
    "test-clean": "wavs/LibriSpeech/test-clean",
    "thchs30": "wavs/data_thchs30",
    "train-galaxy": "metadata/galaxy_train.csv",
    "val-galaxy": "metadata/galaxy_val.csv",
    "vad_train-clean-100": "wavs/LibriSpeech/train-clean-100",
    "vad_train-clean-360": "wavs/LibriSpeech/train-clean-360",
    "vad_train-other-500": "wavs/LibriSpeech/train-other-500",
    "vad_dev-clean": "wavs/LibriSpeech/dev-clean",
    "vad_thchs30": "wavs/data_thchs30",
    "musan": "wavs/noise/musan",
    "wham_noise": "wavs/noise/wham_noise",
    "FSD50K": "wavs/noise/FSD50K",
    "ESC-50-master": "wavs/noise/ESC-50-master",
    "ESC-50": "wavs/noise/ESC-50-master",
    "rirs_noises": "wavs/noise/RIRS_NOISES",
    "dns_challenge": "wavs/noise/DNS-Challenge",
}
ERROR_CORPUS_NOT_FOUND = (
    "❌ Corpus not found.\n"
    "Please verify that the corpus name is correct and that it exists on your system.\n\n"
    "To automatically download the corpus, rerun the command with the following flag:\n\n"
    "    soundkit -t your_task -m data -c your_config.yaml data.force_download=true\n\n"
    "Or manually set your_config.yaml with:\n\n"
    "    data:\n"
    "      force_download: true\n"
)

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

def get_wavefiles(
        path_folder: str)-> list:
    """
    Get all wave or flac files in a folder and its subfolders.
    Args:
        path_folder (str): Path to the folder.
    Returns:
        list: List of wave or flac file paths.
    """
    lst = []
    for root, _, files in os.walk(f'{path_folder}'):
        for file in files:
            if re.search(r'(wav$|flac$)', file):
                lst += [os.path.join(root, file.strip())]
    return lst

# === Train/Val Corpora ===
def load_train_clean_100(corpus: str) -> list:
    """
    Load LibriSpeech train-clean-100 corpus.
    Args:
        corpus (str): Path to the corpus.
    Returns:
        list: List of wave file paths.
    """
    path = corpus2path_map[corpus['name']]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus path does not exist: {path}. {ERROR_CORPUS_NOT_FOUND}")
    return get_wavefiles("wavs/LibriSpeech/train-clean-100")


def load_train_clean_360(corpus: str) -> list:
    """
    Load LibriSpeech train-clean-360 corpus.
    Args:
        corpus (str): Path to the corpus.
    Returns:
        list: List of wave file paths.
    """
    path = corpus2path_map[corpus['name']]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus path does not exist: {path}. {ERROR_CORPUS_NOT_FOUND}")
    return get_wavefiles("wavs/LibriSpeech/train-clean-360")


def load_dev_clean(corpus: str) -> list:
    """
    Load LibriSpeech dev-clean corpus.
    Args:
        corpus (str): Path to the corpus.
    Returns:
        list: List of wave file paths.
    """
    path = corpus2path_map[corpus['name']]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus path does not exist: {path}. {ERROR_CORPUS_NOT_FOUND}")

    return get_wavefiles("wavs/LibriSpeech/dev-clean")


def load_test_clean(corpus: str) -> list:
    """
    Load LibriSpeech test-clean corpus.
    Args:
        corpus (str): Path to the corpus.
    Returns:
        list: List of wave file paths.
    """
    path = corpus2path_map[corpus['name']]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus path does not exist: {path}. {ERROR_CORPUS_NOT_FOUND}")

    return get_wavefiles("wavs/LibriSpeech/test-clean")

def load_thchs30(corpus: str) -> dict:
    """
    Load THCHS-30 corpus.
    Args:
        corpus (str): Path to the corpus.
    Returns:
        dict: Dictionary with 'train' and 'val' keys containing lists of wave file paths.
    """
    path = corpus2path_map[corpus['name']]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus path does not exist: {path}. {ERROR_CORPUS_NOT_FOUND}")

    train_list = get_wavefiles("wavs/data_thchs30/train")
    dev_list = get_wavefiles("wavs/data_thchs30/dev")
    return {"train": train_list, "val": dev_list}

# === KWS Corpora ===
def load_train_galaxy(corpus: str) -> list:
    """
    Load Galaxy corpus for training.
    Args:
        corpus (str): Path to the corpus.
    Returns:
        list: List of wave file paths.
    """
    path = corpus2path_map[corpus['name']]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Due to licensing restrictions, the dataset cannot be downloaded automatically. Please visit https://www.qualcomm.com/developer/software/keyword-speech-dataset/downloads to obtain the dataset.")

    return load_wav_label_csv('metadata/galaxy_train.csv')

def load_val_galaxy(corpus: str) -> list:
    """
    Load Galaxy corpus for validation.
    Args:
        corpus (str): Path to the corpus.
    Returns:
        list: List of wave file paths.
    """
    path = corpus2path_map[corpus['name']]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Due to licensing restrictions, the dataset cannot be downloaded automatically. Please visit https://www.qualcomm.com/developer/software/keyword-speech-dataset/downloads to obtain the dataset.")


    return load_wav_label_csv('metadata/galaxy_val.csv')

# for vad
def load_vad_train_clean_100(corpus: str) -> list:
    """
    Load LibriSpeech train-clean-100 corpus for VAD.
    Args:
        corpus (str): Path to the corpus.
    Returns:
        list: List of wave file paths.
    """
    path = corpus2path_map[corpus['name']]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus path does not exist: {path}. {ERROR_CORPUS_NOT_FOUND}")

    return load_wav_label_csv('metadata/vad/libri_train_clean_100.csv')

def load_vad_train_clean_360(corpus: str) -> list:
    """
    Load LibriSpeech train-clean-360 corpus for VAD.
    Args:
        corpus (str): Path to the corpus.
    Returns:
        list: List of wave file paths.
    """
    path = corpus2path_map[corpus['name']]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus path does not exist: {path}. {ERROR_CORPUS_NOT_FOUND}")

    return load_wav_label_csv('metadata/vad/libri_train_clean_360.csv')

def load_vad_train_other_500(corpus: str) -> list:
    """
    Load LibriSpeech train-other-500 corpus for VAD.
    Args:
        corpus (str): Path to the corpus.
    Returns:
        list: List of wave file paths.
    """
    path = corpus2path_map[corpus['name']]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus path does not exist: {path}. {ERROR_CORPUS_NOT_FOUND}")

    return load_wav_label_csv('metadata/vad/libri_train_other_500.csv')


def load_vad_dev_clean(corpus: str) -> list:
    """
    Load LibriSpeech dev-clean corpus for VAD.
    Args:
        corpus (str): Path to the corpus.
    Returns:
        list: List of wave file paths.
    """
    path = corpus2path_map[corpus['name']]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus path does not exist: {path}. {ERROR_CORPUS_NOT_FOUND}")

    return load_wav_label_csv('metadata/vad/libri_dev_clean.csv')

def load_vad_thchs30(corpus: str) -> dict:
    """
    Load THCHS-30 corpus for VAD.
    Args:
        corpus (str): Path to the corpus.
    Returns:
        dict: Dictionary with 'train' and 'val' keys containing lists of wave file paths.
    """
    path = corpus2path_map[corpus['name']]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus path does not exist: {path}. {ERROR_CORPUS_NOT_FOUND}")

    train_list = load_wav_label_csv('metadata/vad/thchs30_train.csv')
    dev_list = load_wav_label_csv('metadata/vad/thchs30_dev.csv')

    return {"train": train_list, "val": dev_list}

# === Noise Corpora ===

def load_wham_noise(corpus: str) -> dict:
    """
    Load WHAM noise corpus.
    Args:
        corpus (str): Path to the corpus.
    Returns:
        dict: Dictionary with 'train' and 'val' keys containing lists of wave file paths.
    """

    path = corpus2path_map[corpus['name']]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus path does not exist: {path}. {ERROR_CORPUS_NOT_FOUND}")

    files = {}
    files['train'] = get_wavefiles('wavs/noise/wham_noise/tr')
    files['val'] = get_wavefiles('wavs/noise/wham_noise/cv')
    return files

def load_dns_challenge_noise(corpus: str) -> dict:
    """
    Load DNS-Challenge noise corpus.
    Args:
        corpus (str): Path to the corpus.
    Returns:
        dict: Dictionary with 'train' and 'val' keys containing lists of wave file paths.
    """

    path = corpus2path_map[corpus['name']]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus path does not exist: {path}. {ERROR_CORPUS_NOT_FOUND}")

    wavs = get_wavefiles('wavs/noise/DNS-Challenge')
    random.shuffle(wavs)
    split = len(wavs) // 5

    files = {}
    files['train'] = wavs[split:]
    files['val'] = wavs[:split]
    return files

def load_fsd50k(corpus: str) -> dict:
    """
    Load FSD50K corpus.
    Args:
        corpus (str): Path to the corpus.
    Returns:
        dict: Dictionary with 'train' and 'val' keys containing lists of wave file paths.
    """

    path = corpus2path_map[corpus['name']]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus path does not exist: {path}. {ERROR_CORPUS_NOT_FOUND}")

    with open('metadata/FSD50K/non_speech.csv', 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    random.shuffle(lines)
    split = len(lines) // 5
    return {"train": lines[split:], "val": lines[:split]}


def load_esc50(corpus: str) -> dict:
    """
    Load ESC-50 corpus.
    Args:
        corpus (str): Path to the corpus.
    Returns:
        dict: Dictionary with 'train' and 'val' keys containing lists of wave file paths.
    """
    path = corpus2path_map[corpus['name']]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus path does not exist: {path}. {ERROR_CORPUS_NOT_FOUND}")


    with open('metadata/ESC-50-master/non_speech.csv', 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    random.shuffle(lines)
    split = len(lines) // 5
    return {"train": lines[split:], "val": lines[:split]}


def load_musan(corpus: str) -> dict:
    """
    Load MUSAN corpus.
    Args:
        corpus (str): Path to the corpus.
    Returns:
        dict: Dictionary with 'train' and 'val' keys containing lists of wave file paths.
    """
    path = corpus2path_map[corpus['name']]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus path does not exist: {path}. {ERROR_CORPUS_NOT_FOUND}")


    music = get_wavefiles('wavs/noise/musan/music')
    noise = get_wavefiles('wavs/noise/musan/noise')
    lines = music + noise
    random.shuffle(lines)
    split = len(lines) // 5
    return {"train": lines[split:], "val": lines[:split]}

# === Reverb Corpus ===

def load_rirs_noises(corpus: str) -> dict:
    """
    Load RIRS_NOISES corpus.
    Args:
        corpus (str): Path to the corpus.
    Returns:
        dict: Dictionary with 'train' and 'val' keys containing lists of wave file paths.
    """
    path = corpus2path_map[corpus['name']]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus path does not exist: {path}. {ERROR_CORPUS_NOT_FOUND}")

    all_files = get_wavefiles('wavs/noise/RIRS_NOISES')
    random.shuffle(all_files)
    split = len(all_files) // 5
    return {"train": all_files[split:], "val": all_files[:split]}
