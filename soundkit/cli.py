import os
from pathlib import Path
from argdantic import ArgField, ArgParser
from omegaconf import OmegaConf, DictConfig
from .defines import SKTaskParams, SKMode
from .tasks import TaskFactory  # assume this exists
import soundkit.datasets.register_datasets  # assume this exists
parser = ArgParser()

def parse_config(path: str) -> DictConfig:
    """
    Load and resolve config file as OmegaConf DictConfig.
    Uses SKTaskParams as the schema for validation.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = OmegaConf.load(path)
    schema = OmegaConf.structured(SKTaskParams)
    cfg = OmegaConf.merge(schema, raw)
    OmegaConf.resolve(cfg)
    return cfg

@parser.command()
def run(
    mode: str = ArgField("-m", default=SKMode.train),
    task: str = ArgField("-t", default="se"),
    config: str = ArgField("-c", default="./configs/se.yaml"),
    tensorboard: bool = ArgField("--tensorboard", default=False),
):
    """SoundKit CLI entry point."""
    print(f"🔧 Mode: {mode}, Task: {task}")

    params = parse_config(config)
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
            task_handler.demo(params)

def main():
    """Main entry point for the SoundKit CLI."""
    parser()

if __name__ == "__main__":
    main()
