''' Soundkit CLI '''
import os
from pathlib import Path
from typing import Type, TypeVar
from argdantic import ArgField, ArgParser
from pydantic import BaseModel, ValidationError
from omegaconf import OmegaConf
from .defines import SKTaskParams, SKMode
from .tasks import TaskFactory
B = TypeVar("B", bound=BaseModel)
parser=ArgParser()

def parse_content(cls: Type[B], content: str) -> B:
    """
    Parse YAML config file using OmegaConf and validate with a Pydantic model.

    Args:
        cls (B): Pydantic model subclass
        content (str): File path to YAML file

    Returns:
        B: Validated instance of the Pydantic model

    Raises:
        FileNotFoundError: If file is not found
        ValueError: For YAML parsing or validation errors
    """
    try:
        cfg = OmegaConf.load(content)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {content}")
    except Exception as e:
        raise ValueError(f"YAML parsing failed: {e}")

    try:
        return cls.model_validate(OmegaConf.to_container(cfg, resolve=True))
    except ValidationError as e:
        raise ValueError(f"Config validation failed for {cls.__name__}:\n{e}")


@parser.command()
def run(
        mode: str = ArgField("-m", default=SKMode.train),
        task: str = ArgField("-t", default="se"),
        config: str = ArgField("-c", default="./configs/se.yaml"),
        tensorboard: bool = ArgField("--tensorboard", default=False),
    ):
    """SoundKit CLI entry point.

    Args:
        mode (SKMode, optional): Mode. Defaults to SKMode.train.
        task (str, optional): Task. Defaults to "rhythm".
        config (str, optional): File path or YAML content. Defaults to "{}".
    """
    print(f"Mode: {mode}, Task: {task}")

    params = parse_content(SKTaskParams, config)
    task_handler = TaskFactory.get(task)

    match mode:
        case SKMode.data:
            task_handler.data(params)

        case SKMode.train:
            if tensorboard:
                tb_dir = params.train['path']['tensorboard_dir']

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
    ''' Main function to run the CLI '''
    parser()

if __name__ == "__main__":
    main()
