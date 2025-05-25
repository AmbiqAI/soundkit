import sys
import tempfile
from pydantic import BaseModel, Field
from typing import Any
from pathlib import Path
from enum import Enum

# Backport StrEnum for Python < 3.11
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    class StrEnum(str, Enum):
        pass

class NamedParams(BaseModel, extra="allow"):
    """
    Named parameters is used to store parameters for a specific model, preprocessing, or augmentation.
    Typically name refers to class/method name and params is provided as kwargs.

    Attributes:
        name: Name
        params: Parameters
    """

    name: str = Field(..., description="Name")
    params: dict[str, Any] = Field(default_factory=dict, description="Parameters")

class SKTaskParams(BaseModel, extra="allow"):
    """SoundKit Task configuration params"""
    # Common arguments
    name: str = Field("experiment", description="Experiment name")
    project: str = Field("soundkit", description="Project name")
    job_dir: Path = Field(
        default_factory=lambda: Path(tempfile.gettempdir()),
        description="Job output directory",
    )

    data: dict[str, Any] = Field(
        default_factory=dict, description="Data parameters")

    # Training arguments
    train: dict[str, Any] = Field(
        default_factory=dict, description="training parameters")

    # Evaluating arguments
    evaluate: dict[str, Any] = Field(
        default_factory=dict, description="evaluating parameters")
    
    # export arguments
    export: dict[str, Any] = Field(
        default_factory=dict, description="exporting parameters")

    # demo arguments
    demo: dict[str, Any] = Field(
        default_factory=dict, description="demo parameters")

class SKMode(StrEnum):
    """SoundKit task mode"""

    data = "data"
    train = "train"
    evaluate = "evaluate"
    export = "export"
    demo = "demo"
