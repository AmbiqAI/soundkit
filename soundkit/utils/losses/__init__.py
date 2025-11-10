"""Losses module for soundkit."""
from typing import Type, Any
import tensorflow as tf
from soundkit.utils.WarmUpCosineDecay import WarmUpCosineDecay
from .loss_utils import (FramewiseMSE,
                         FramewiseMAE,
                         CompressedMSE,
                         LogFramewiseMSE,
                         SISDRLoss)
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
    def get(cls, name: str, params: Any = None):
        """Get a loss function by name, optionally with params."""
        if name not in cls._registry:
            raise ValueError(f"Loss '{name}' is not registered.")
        params = params or {}
        return cls._registry[name](**params)

# Register classes directly
LossFactory.register("mse", FramewiseMSE)
LossFactory.register("mae", FramewiseMAE)
LossFactory.register("log_mse", LogFramewiseMSE)
LossFactory.register("compressed_mse", CompressedMSE)
LossFactory.register("mrl_mse", MultiResolutionSTFTLossFromSTFT)
LossFactory.register("focal", FocalLoss)
LossFactory.register("si_sdr", SISDRLoss)
LossFactory.register("cross_entropy", tf.keras.losses.SparseCategoricalCrossentropy)
