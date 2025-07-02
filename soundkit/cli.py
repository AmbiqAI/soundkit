import os
import sys
import subprocess
import argparse
from pathlib import Path
from argdantic import ArgField, ArgParser
from omegaconf import OmegaConf, DictConfig
from .defines import SKTaskParams, SKMode
def install_missing_packages():
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torchmetrics", "librosa", "onnxruntime-gpu", "requests"
    ])

try:
    import torch
    import torchaudio
    import torchmetrics
except ImportError:
    install_missing_packages()
    import torch
    import torchaudio
    import torchmetrics

from .tasks import TaskFactory

# === Global to pass dotlist overrides ===
extra_overrides: list[str] = []

parser = ArgParser()

def parse_config(path: str, overrides: list[str] = None) -> DictConfig:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = OmegaConf.load(path)
    schema = OmegaConf.structured(SKTaskParams)
    cfg = OmegaConf.merge(schema, raw)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    OmegaConf.resolve(cfg)
    return cfg

# === Real logic (can be called from anywhere) ===
def run_task(mode: str, task: str, config: str, tensorboard: bool, view: bool = False):
    print(f"🔧 Mode: {mode}, Task: {task}")
    print(f"🛠️  Overrides: {extra_overrides}")

    params = parse_config(config, overrides=extra_overrides)
    task_handler = TaskFactory.get(task)

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
    
                if params.demo.platform == "evb":
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

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", type=str)
    ap.add_argument("-t", "--task", type=str)
    ap.add_argument("-c", "--config", type=str)
    ap.add_argument("--tensorboard", action="store_true")
    ap.add_argument("--view", action="store_true")

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
