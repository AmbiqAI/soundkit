import os
import logging
import argparse
from pathlib import Path
from argdantic import ArgField, ArgParser
from omegaconf import OmegaConf, DictConfig
import boto3
from .defines import SKTaskParams, SKMode

from .tasks import TaskFactory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
log = logging.getLogger(__name__)

# === Global to pass dotlist overrides ===
extra_overrides: list[str] = []

parser = ArgParser()

def parse_config(
        path: str,
        overrides: list[str] = None) -> DictConfig:
    """ 
    Parse YAML configuration file with optional overrides. 
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = OmegaConf.load(path)
    schema = OmegaConf.structured(SKTaskParams)
    cfg = OmegaConf.merge(schema, raw)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    OmegaConf.set_struct(cfg, True)  # <--- Enforce struct mode for safety
    OmegaConf.resolve(cfg)
    OmegaConf.to_object(cfg)


    return cfg

def download_s3_folder(bucket_name, s3_prefix, local_dir):
    s3 = boto3.client('s3')
    local_dir = Path(local_dir)
    
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' not in result:
            log.warning(f"⚠️ No objects found in S3 with prefix: {s3_prefix}")
            continue
            
        for obj in result['Contents']:
            key = obj['Key']
            
            # Extract relative path from S3 key
            # If key is 'soundkit/id/id.yaml' and prefix is 'soundkit/id/'
            # relative_path becomes 'id.yaml'
            relative_path = key[len(s3_prefix):].lstrip('/')
            if not relative_path: 
                continue
            
            local_file_path = local_dir / relative_path
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            log.info(f"  ⬇️ {relative_path}")
            s3.download_file(bucket_name, key, str(local_file_path))

# === Real logic (can be called from anywhere) ===
def run_task(
        mode: str,
        task: str,
        config: str,
        tensorboard: bool,
        view: bool = False):

    print(f"🔧 Mode: {mode}, Task: {task}")
    print(f"🛠️  Overrides: {extra_overrides}")

    # 1. Normalize the path to handle './zoo', 'zoo/', or '../zoo'
    # .resolve() makes the path absolute and removes things like './'
    absolute_config_path = Path(config).resolve()
    
    # 2. Check if 'zoo' is in the folder structure AND it doesn't exist locally
    if "zoo" in absolute_config_path.parts and not Path(config).exists():
        log.info(f"📂 Detected Model Zoo path: {config}")
        log.info(f"📥 Config not found locally. Syncing from S3...")
        
        bucket = "ambiqai-model-zoo"
        # We assume the S3 structure is soundkit/task_name
        s3_folder = f"soundkit/{task}/"
        
        # Determine the local target directory (e.g., ./zoo/id/)
        # This finds where 'zoo' starts in your input path
        zoo_index = absolute_config_path.parts.index("zoo")
        local_target = Path(*absolute_config_path.parts[:zoo_index + 2])
        
        try:
            log.info(f"🔄 Syncing S3 folder s3://{bucket}/{s3_folder} to {local_target}...")
            download_s3_folder(bucket, s3_folder, local_target)
        except Exception as e:
            log.error(f"❌ S3 Sync failed: {e}")
            return

    params = parse_config(config, overrides=extra_overrides)
    task_handler = TaskFactory.get(task)
    if task == "id":
        if params.train.batchsize != params.data.ppls_per_group * params.data.num_sentences:
            params.train.batchsize = params.data.ppls_per_group * params.data.num_sentences
            log.warning(
                f"Adjusted train batchsize to {params.train.batchsize} based on ppls_per_group and num_sentences."
                )
    match mode:
        case SKMode.data:
            task_handler.data(params)
        case SKMode.train:
            if tensorboard:
                tb_dir = params.train["path"]["tensorboard_dir"]
                parent_tb_dir = str(Path(tb_dir).parent)
                print(f"🚀 Launching TensorBoard at: {parent_tb_dir}")
                os.system(f"tensorboard --logdir={parent_tb_dir}")
            else:
                task_handler.train(params)
        case SKMode.evaluate:
            task_handler.evaluate(params)
        case SKMode.export:
            task_handler.export(params)
        case SKMode.demo:
            if view:
                if task in ["se", "vad", "kws"] :
                    script="audioview_se"
                elif task == "id":
                    script="audioview_nnid"

                print("🔌 Running EVB demo...")

                os.system(f"python -m soundkit.tools.{script}")
            else:
                task_handler.demo(params)

# === CLI entry point for argdantic users ===
@parser.command()
def run_cli(
    mode: str = ArgField("-m", default=SKMode.train),
    task: str = ArgField("-t", default="se"),
    config: str = ArgField("-c", default="./configs/se.yaml"),
    tensorboard: bool = ArgField("--tensorboard", default=False),
    view: bool = ArgField("--view", default=False),
):
    run_task(mode, task, config, tensorboard, view)

# === Main entrypoint for manual CLI invocation ===
def main():
    global extra_overrides

    ap = argparse.ArgumentParser(
        description="SoundKit CLI - data, train, evaluate, export, and demo AI tasks.",
    )

    ap.add_argument(
        "-t", "--task",
        type=str,
        help="Task name: se, vad, kws, id")
    ap.add_argument(
        "-m", "--mode",
        type=str,
        help="Execution mode: data, train, evaluate, export, demo")
    ap.add_argument(
        "-c", "--config",
        type=str,
        help="Path to YAML configuration file")
    ap.add_argument(
        "--tensorboard",
        action="store_true",
        help="Launch TensorBoard (only for train mode)")
    ap.add_argument(
        "--view",
        action="store_true",
        help="Enable waveform/audio viewer (only for demo)")

    known_args, unknown_args = ap.parse_known_args()
    extra_overrides = unknown_args

    # 🚀 Call the logic directly — no CLI decorator involved
    run_task(
        mode=known_args.mode or SKMode.train,
        task=known_args.task or "se",
        config=known_args.config or "./configs/se.yaml",
        tensorboard=known_args.tensorboard,
        view=known_args.view
    )

if __name__ == "__main__":
    main()
