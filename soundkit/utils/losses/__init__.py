"""Losses module for soundkit."""
from typing import Type
from .loss_utils import FramewiseMSE, FramewiseMAE, CompressedMSE

class LossFactory:
    """Factory class for creating loss functions."""
    _registry = {}

    @classmethod
    def register(cls, name: str, loss_cls: Type):
        """Register a loss function with a name."""
        cls._registry[name] = loss_cls

    @classmethod
    def get(cls, name: str, **kwargs):
        """Get a loss function by name."""
        if name not in cls._registry:
            raise ValueError(f"Loss '{name}' is not registered.")
        return cls._registry[name](**kwargs)

# Register classes directly
LossFactory.register("mse", FramewiseMSE)
LossFactory.register("mae", FramewiseMAE)
LossFactory.register("compressed_mse", CompressedMSE)
