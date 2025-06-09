"""Losses module for soundkit."""
from typing import Type
import tensorflow as tf
from .loss_utils import FramewiseMSE, FramewiseMAE, CompressedMSE

from .loss_mrl import MultiResolutionSTFTLossFromSTFT
from .loss_focal import FocalLoss


class LossFactory:
    """Factory class for creating loss functions."""
    _registry = {}

    @classmethod
    def register(cls, name: str, loss_cls: Type):
        """Register a loss function with a name."""
        cls._registry[name] = loss_cls

    @classmethod
    def get(cls, name: str, params: dict = None):
        """Get a loss function by name, optionally with params."""
        if name not in cls._registry:
            raise ValueError(f"Loss '{name}' is not registered.")
        params = params or {}
        return cls._registry[name](**params)


# Register classes directly
LossFactory.register("mse", FramewiseMSE)
LossFactory.register("mae", FramewiseMAE)
LossFactory.register("compressed_mse", CompressedMSE)
LossFactory.register("mrl_mse", MultiResolutionSTFTLossFromSTFT)
LossFactory.register("focal", FocalLoss)
LossFactory.register("cross_entropy", tf.keras.losses.SparseCategoricalCrossentropy)
