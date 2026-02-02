"""
Download the required SE training dataset
"""
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import tarfile
import zipfile
import requests
import time
from tqdm import tqdm

DOWNLOADABLE_CORPORA = {
    "train-clean-100",
    "train-clean-360",
    "dev-clean",
    "train-other-500",
    "vad_train-clean-100",
    "vad_train-clean-360",
    "vad_dev-clean",
    "vad_train-other-500",
    "thchs30",
    "vad_thchs30",
    "musan",
    "wham_noise",
    "ESC-50-master",
    "ESC-50",
    "rirs_noises",
    "FSD50K",
}
QUALCOMM_KWS_CORPORA = {
    "train-galaxy",
    "val-galaxy",
    "train-coros",
    "val-coros",
}
QUALCOMM_KWS_URL = (
    "https://softwarecenter.qualcomm.com/api/download/software/dataset/"
    "AIDataset/Qualcomm_Keyword_Speech_Dataset/qualcomm_keyword_speech_dataset.zip"
)
QUALCOMM_KWS_ARCHIVE = "qualcomm_keyword_speech_dataset.zip"
QUALCOMM_KWS_METADATA = {
    "galaxy_train.csv",
    "galaxy_val.csv",
    "coros_train.csv",
    "coros_val.csv",
}

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
        # Choose correct tarfile mode
        if re.search(r'\.(tar\.gz|tgz)$', zip_file_path):
            tar_mode = "r:gz"
        elif re.search(r'\.tar\.bz2$', zip_file_path):
            tar_mode = "r:bz2"
        elif re.search(r'\.tar\.xz$', zip_file_path):
            tar_mode = "r:xz"
        else:
            tar_mode = "r"
        with tarfile.open(zip_file_path, tar_mode) as tar:
            members = tar.getmembers()
            total_size = sum(member.size for member in members)

            with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                for member in members:
                    tar.extract(member, extract_to)
                    pbar.update(member.size)

def url_download(
        url: str,
        target_name: str,
        user_agent: str | None = None,
        retries: int = 3,
        timeout: int = 30) -> None:
    """
    download file from url
    """
    headers = {"User-Agent": user_agent} if user_agent else {}
    block_size = 1024 * 1024  # 1 MiB

    for attempt in range(1, retries + 1):
        existing_size = 0
        mode = "wb"
        if os.path.exists(target_name):
            existing_size = os.path.getsize(target_name)
            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"
                mode = "ab"

        try:
            response = requests.get(url, stream=True, headers=headers, timeout=timeout)
            print(f"Downloading {url} (attempt {attempt}/{retries})")
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            if response.status_code == 200 and existing_size > 0:
                # Server ignored Range; restart download.
                mode = "wb"
                existing_size = 0

            with tqdm(
                total=total_size + existing_size if total_size else None,
                unit="B",
                unit_scale=True,
                initial=existing_size,
            ) as progress_bar:
                with open(target_name, mode) as file:
                    for data in response.iter_content(block_size):
                        if not data:
                            continue
                        progress_bar.update(len(data))
                        file.write(data)
            return
        except (requests.RequestException, RuntimeError) as exc:
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                if exc.response.status_code == 416 and os.path.exists(target_name):
                    # Requested range not satisfiable; restart from scratch.
                    print("Server rejected Range request; restarting download from scratch.")
                    try:
                        os.remove(target_name)
                    except OSError:
                        pass
                    if "Range" in headers:
                        headers.pop("Range", None)
                    continue
            if attempt >= retries:
                raise
            wait_s = 2 ** attempt
            print(f"Download error: {exc}. Retrying in {wait_s}s...")
            time.sleep(wait_s)

def _copy_qualcomm_metadata(search_root: str, metadata_dir: str = "metadata") -> None:
    os.makedirs(metadata_dir, exist_ok=True)
    if all(os.path.exists(os.path.join(metadata_dir, fname)) for fname in QUALCOMM_KWS_METADATA):
        # Metadata already provided (e.g., synced from S3).
        return
    found = 0
    for root, _, files in os.walk(search_root):
        for fname in files:
            if fname in QUALCOMM_KWS_METADATA:
                src = os.path.join(root, fname)
                dst = os.path.join(metadata_dir, fname)
                shutil.copyfile(src, dst)
                found += 1
    if found == 0:
        print(
            "Warning: Qualcomm metadata CSVs were not found after extraction. "
            "Expected galaxy/coros CSVs in the archive."
        )

def corpus_download(
        corpus: str,
        type_cropus: str = 'noise',
        accept_qualcomm_license: bool = False,
        ) -> None:
    """
    download se dataset
    """
    if auth_token:
        print(f"--> Downloading {corpus} using provided Qualcomm token...")
        # Your logic here to hit the gated URL using the token
        # e.g., requests.get(url, headers={"Authorization": f"Bearer {auth_token}"})
    else:
        print(f"--> Downloading {corpus} from public source...")
        # Your existing public download logic
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

        case (
          'train-clean-360'
        | 'vad_train-clean-360' # for vad
        | 'train-clean-100'
        | 'vad_train-clean-100' # for vad
        | 'dev-clean'
        | 'vad_dev-clean' # for vad
        | 'train-other-500'
        | 'vad_train-other-500' # for vad
        ):
            if corpus == 'vad_train-clean-360':
                corpus = 'train-clean-360'
            elif corpus == 'vad_train-clean-100':
                corpus = 'train-clean-100'
            elif corpus == 'vad_dev-clean':
                corpus = 'dev-clean'
            elif corpus == 'vad_train-other-500':
                corpus = 'train-other-500'
            target_name = f'{corpus}.tar.gz'
            target_path = f'./{tmp_download}/{target_name}'
            url = f'https://www.openslr.org/resources/12/{target_name}'
            url_download(url, target_path )
            unzip_with_progress(target_path, dst_folder)

        case (
          'qualcomm_keyword_speech_dataset'
        | 'train-galaxy'
        | 'val-galaxy'
        | 'train-coros'
        | 'val-coros'
        ):
            if not accept_qualcomm_license:
                raise PermissionError(
                    "Qualcomm Keyword Speech Dataset requires license acceptance. "
                    "Set data.accept_qualcomm_license=true to enable automatic download."
                )
            dst_folder = "./wavs/kws"
            os.makedirs(dst_folder, exist_ok=True)
            target_path = f'./{tmp_download}/{QUALCOMM_KWS_ARCHIVE}'
            url_download(QUALCOMM_KWS_URL, target_path)
            unzip_with_progress(target_path, dst_folder)
            _copy_qualcomm_metadata(dst_folder)

        case 'thchs30' | 'vad_thchs30':  # for vad
            if corpus == 'vad_thchs30':
                corpus = 'thchs30'
            target_name = f'{corpus}.tgz'
            target_path = f'./{tmp_download}/{target_name}'
            url = f'https://openslr.trmal.net/resources/18/data_thchs30.tgz'
            url_download(url, target_path )
            unzip_with_progress(target_path, dst_folder)

        case 'musan':
            target_name = f'{corpus}.tar.gz'
            target_path = f'./{tmp_download}/{target_name}'
            url = f'https://www.openslr.org/resources/17/{target_name}'
            url_download(url, target_path )
            unzip_with_progress(target_path, dst_folder)

        case 'wham_noise':
            target_name = f'{corpus}.zip'
            target_path = f'./{tmp_download}/{target_name}'
            url = f'https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/{target_name}'
            url_download(url, target_path )
            unzip_with_progress(target_path, dst_folder)

        case 'ESC-50-master' | 'ESC-50':
            target_name = f'master.zip'
            target_path = f'./{tmp_download}/{target_name}'
            url = f'https://github.com/karoldvl/ESC-50/archive/{target_name}'
            url_download(url, target_path )
            unzip_with_progress(target_path, dst_folder)
            shutil.copyfile(
                'metadata/ESC-50-master/non_speech.csv',
                'wavs/noise/ESC-50-master/non_speech.csv')

        case 'rirs_noises':
            target_name = f'{corpus}.zip'
            target_path = f'./{tmp_download}/{target_name}'
            url = f'https://www.openslr.org/resources/28/{target_name}'
            url_download(url, target_path )
            unzip_with_progress(target_path, dst_folder)

        case "val-galaxy" | "train-galaxy":
            target_name = f'qualcomm_keyword_speech_dataset.zip'
            target_path = f'./{tmp_download}/{target_name}'
            url = f'https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Qualcomm_Keyword_Speech_Dataset/qualcomm_keyword_speech_dataset.zip'
            url_download(url, target_path )
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
            user_agent = "soundkit-fsd50k-download/1.0"
            max_workers = 3
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for fname in fsd50_lst:
                    url = f'https://zenodo.org/records/4060432/files/{fname}?download=1'
                    futures.append(
                        executor.submit(
                            url_download,
                            url,
                            f"./{tmp_download}/{fname}",
                            user_agent,
                        )
                    )
                for future in as_completed(futures):
                    future.result()
            os.system(f"zip -s 0 ./{tmp_download}/FSD50K.dev_audio.zip --out ./{tmp_download}/unsplit.zip")
            unzip_with_progress(f"./{tmp_download}/unsplit.zip", "./wavs/noise/FSD50K/")

            shutil.copyfile(
                'metadata/FSD50K/non_speech.csv',
                'wavs/noise/FSD50K/non_speech.csv'
                )
