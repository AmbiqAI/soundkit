# soundkit/utils/s3.py
import logging
from pathlib import Path
from tqdm import tqdm

import boto3
from botocore import UNSIGNED
from botocore.config import Config

log = logging.getLogger(__name__)

class S3Manager:
    def __init__(self, bucket_name="ambiqai-model-zoo", s3_config: Config | None = None):
        self.bucket_name = bucket_name
        # Default to unsigned requests for public buckets to avoid credential lookup.
        if s3_config is None:
            s3_config = Config(signature_version=UNSIGNED)
        self.s3_client = boto3.client("s3", config=s3_config)

    def download_folder(self, s3_prefix: str, local_dir: Path):
        """Recursively downloads a folder from S3 with a progress bar."""
        local_dir.mkdir(parents=True, exist_ok=True)
        
        paginator = self.s3_client.get_paginator('list_objects_v2')
        objects = []
        
        # 1. Collect all objects to calculate total size/count for progress bar
        for result in paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix):
            if 'Contents' in result:
                objects.extend(result['Contents'])

        if not objects:
            log.warning(f"⚠️ No objects found in S3 with prefix: {s3_prefix}")
            return

        # 2. Download with progress bar
        for obj in tqdm(objects, desc=f"Syncing {s3_prefix}", unit="file"):
            key = obj['Key']
            relative_path = key[len(s3_prefix):].lstrip('/')
            if not relative_path: continue
            
            local_file_path = local_dir / relative_path
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.s3_client.download_file(self.bucket_name, key, str(local_file_path))

def sync_metadata(local_root: Path = Path(".")):
    """Convenience function to ensure metadata is present."""
    manager = S3Manager()
    local_path = local_root / "metadata"
    if not local_path.exists():
        log.info("📂 Metadata folder not found. Downloading...")
        manager.download_folder("soundkit/metadata/", local_path)
