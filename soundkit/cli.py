import os
import logging
import argparse
from pathlib import Path
from argdantic import ArgField, ArgParser
from omegaconf import OmegaConf, DictConfig
from soundkit.utils.s3 import (
        S3Manager,
        sync_metadata
    )
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

# === Real logic (can be called from anywhere) ===
def run_task(
        mode: str,
        task: str,
        config: str,
        tensorboard: bool,
        view: bool = False):

    print(f"🔧 Mode: {mode}, Task: {task}")
    print(f"🛠️  Overrides: {extra_overrides}")

    # Always ensure metadata is present first
    sync_metadata()

    config_path = Path(config)
    abs_config_path = config_path.resolve()
    # Handle the Model Zoo 'zoo' logic
    if "zoo" in abs_config_path.parts:
        zoo_idx = abs_config_path.parts.index("zoo")
        # local_target becomes /your/path/zoo/task_name/
        local_target = Path(*abs_config_path.parts[:zoo_idx + 2])
        
        # Use the modulized manager
        s3 = S3Manager()
        s3.download_folder(s3_prefix=f"soundkit/{task}/", local_dir=local_target)

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
