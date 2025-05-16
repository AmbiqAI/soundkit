"""
Download the required SE training dataset
"""
import os
import re
import shutil
import tarfile
import zipfile
import requests
from tqdm import tqdm

def unzip_with_progress(
        zip_file_path: str,
        extract_to: str) -> None:
    """
    extract zip or tar file
    """
    print(f"Extracting {zip_file_path} to {extract_to}")

    if re.search(r'\.zip$', zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            total_size = sum(file.file_size for file in zip_ref.infolist())
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Unzipping") as pbar:
                for file in zip_ref.infolist():
                    zip_ref.extract(file, extract_to)
                    pbar.update(file.file_size)
    else:
        with tarfile.open(zip_file_path, "r") as tar:
            members = tar.getmembers()
            total_size = sum(member.size for member in members)

            with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                for member in members:
                    tar.extract(member, extract_to)
                    pbar.update(member.size)

def url_download(
        url: str,
        target_name: str) -> None:
    """
    download file from url
    """
    response = requests.get(url, stream=True)
    print(f"Downloading {url}")
    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte 
    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(target_name, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")

def corpus_download(
        corpus: str,
        type_cropus: str = 'noise',
        ) -> None:
    """
    download se dataset
    """
    wavs = "wavs"
    tmp_download = "tmp"
    os.makedirs(wavs, exist_ok=True)
    os.makedirs(tmp_download, exist_ok=True)
    if type_cropus in ('noise', 'reverb'):
        os.makedirs(f"{wavs}/noise", exist_ok=True)
        dst_folder = f'./{wavs}/noise/'
    else:
        dst_folder = f'./{wavs}/'

    match corpus:

        case 'train-clean-360' | 'train-clean-100' | 'dev-clean':
            target_name = f'{corpus}.tar.gz'
            target_path = f'./{tmp_download}/{target_name}'
            url = f'https://us.openslr.org/resources/12/{target_name}'
            url_download(url, target_path )
            unzip_with_progress(target_path, dst_folder)

        case 'thchs30':
            target_name = f'{corpus}.tgz'
            target_path = f'./{tmp_download}/{target_name}'
            url = f'https://openslr.elda.org/resources/18/{target_name}'
            url_download(url, target_path )
            unzip_with_progress(target_path, dst_folder)

        case 'musan':
            target_name = f'{corpus}.tar.gz'
            target_path = f'./{tmp_download}/{target_name}'
            url = f'https://us.openslr.org/resources/17/{target_name}'
            url_download(url, target_path )
            unzip_with_progress(target_path, dst_folder)

        case 'wham_noise':
            target_name = f'{corpus}.zip'
            target_path = f'./{tmp_download}/{target_name}'
            url = f'https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/{target_name}'
            url_download(url, target_path )
            unzip_with_progress(target_path, dst_folder)

        case 'ESC-50':
            target_name = f'master.zip'
            target_path = f'./{tmp_download}/{target_name}'
            url = f'https://github.com/karoldvl/ESC-50/archive/{target_name}'
            url_download(url, target_path )
            unzip_with_progress(target_path, dst_folder)
            if corpus == 'ESC-50':
                shutil.copyfile(
                    'data/ESC-50-master/non_speech.csv',
                    'wavs/noise/ESC-50-master/non_speech.csv')
            else:
                shutil.copyfile(
                    'data/FSD50K/non_speech.csv',
                    'wavs/noise/FSD50K/non_speech.csv')

        case 'rirs_noises':
            target_name = f'{corpus}.zip'
            target_path = f'./{tmp_download}/{target_name}'
            url = f'https://www.openslr.org/resources/28/{target_name}'
            # url_download(url, target_path )
            unzip_with_progress(target_path, dst_folder)

        case 'FSD50K':
            fsd50_lst = [
                "FSD50K.dev_audio.z01",
                "FSD50K.dev_audio.z02",
                "FSD50K.dev_audio.z03",
                "FSD50K.dev_audio.z04",
                "FSD50K.dev_audio.z05",
                "FSD50K.dev_audio.zip",
            ]
            for fname in fsd50_lst:
                url = f'https://zenodo.org/record/4060432/files/{fname}?download=1"'
                url_download(url, f"../{tmp_download}/{fname}")
            os.system(f"zip -s 0 ./{tmp_download}/FSD50K.dev_audio.zip --out ./{tmp_download}/unsplit.zip")
            unzip_with_progress(f"./{tmp_download}/unsplit.zip", "./wavs/noise/FSD50K/")
            shutil.copyfile(
                'data/non_speech_fsd50k.csv',
                'wavs/noise/FSD50K/non_speech.csv')
