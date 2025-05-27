# soundkit/utils/dataset_registry.py

from typing import Callable, Dict

class DatasetRegistry:
    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(func: Callable):
            cls._registry[name] = func
            return func
        return decorator

    @classmethod
    def get(cls, name: str) -> Callable:
        if name not in cls._registry:
            raise ValueError(f"Dataset '{name}' is not registered.")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list:
        return list(cls._registry.keys())
