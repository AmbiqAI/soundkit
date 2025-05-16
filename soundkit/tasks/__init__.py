from typing import Type
from .task import SKTask
from .se import SeTask

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
