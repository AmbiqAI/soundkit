from typing import Type
from .task import SKTask
from .se import SeTask
from .se2 import Se2Task

from .vad import VadTask
from .vad2 import Vad2Task

from .kws import KwsTask
from .id import IdTask

class TaskFactory:
    _registry = {}

    @classmethod
    def register(cls, name: str, task_cls: Type[SKTask]):
        cls._registry[name] = task_cls

    @classmethod
    def get(cls, name: str) -> SKTask:
        return cls._registry[name]()

    @classmethod
    def list(cls):
        return list(cls._registry.keys())

TaskFactory.register("se", SeTask)
TaskFactory.register("se2", Se2Task)
TaskFactory.register("vad", VadTask)
TaskFactory.register("vad2", Vad2Task)
TaskFactory.register("kws", KwsTask)
TaskFactory.register("id", IdTask)
