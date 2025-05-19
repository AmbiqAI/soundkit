''' Soundkit CLI '''
import os
from typing import Type, TypeVar
from argdantic import ArgField, ArgParser
from pydantic import BaseModel
import yaml
from .defines import SKTaskParams, SKMode
from .tasks import TaskFactory
B = TypeVar("B", bound=BaseModel)
parser=ArgParser()

def parse_content(cls: Type[B], content: str) -> B:
    """Parse file or raw content into a Pydantic model.

    Args:
        cls (B): Pydantic model subclass
        content (str): File path to YAML content

    Returns:
        B: Pydantic model subclass instance

    Raises:
        FileNotFoundError: If the file does not exist
        yaml.YAMLError: If the file cannot be parsed as YAML
        pydantic.ValidationError: If the data doesn't match the schema
    """
    try:
        with open(content, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {content}")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parsing error in file {content}: {e}")

    try:
        return cls.model_validate(data)
    except Exception as e:
        raise ValueError(f"Failed to validate config against {cls.__name__}: {e}")


@parser.command()
def run(
        mode: str = ArgField("-m", default=SKMode.train),
        task: str = ArgField("-t", default="se"),
        config: str = ArgField("-c", default="./soundkit/tasks/se/configs/se.yaml"),
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
