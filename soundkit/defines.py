import sys
import tempfile
from typing import Any
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

# Backport StrEnum for Python < 3.11
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    class StrEnum(str, Enum):
        pass

@dataclass
class NamedParams:
    """Used to describe a named object and its parameters."""
    name: str
    params: dict[str, Any] = field(default_factory=dict)

@dataclass
class SKTaskParams:
    """SoundKit Task configuration parameters"""

    # Common arguments
    name: str = "experiment"
    project: str = "soundkit"
    job_dir: Path = field(default_factory=tempfile.gettempdir)

    # Task-specific arguments
    data: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    evaluate: dict[str, Any] = field(default_factory=dict)
    export: dict[str, Any] = field(default_factory=dict)
    demo: dict[str, Any] = field(default_factory=dict)

class SKMode(StrEnum):
    """SoundKit task mode"""
    data = "data"
    train = "train"
    evaluate = "evaluate"
    export = "export"
    demo = "demo"
