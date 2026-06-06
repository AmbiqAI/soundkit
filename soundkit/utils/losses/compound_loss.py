"""
Compound loss utilities for SoundKit.
Implements CompoundLoss for combining multiple loss functions with weights, and a factory for config-driven instantiation.
"""
import tensorflow as tf
import collections.abc
from soundkit.utils.losses.loss_mrl import MultiResolutionSTFTLossFromSTFT
from soundkit.utils.losses.loss_utils import (
    SISDRLoss,
    FramewiseMSE,
    FramewiseMAE,
    LogFramewiseMSE,
    CompressedMSE,
)
class CompoundLoss(tf.keras.losses.Loss):
    """
    Compound loss that combines multiple loss functions with specified weights.
    Supports config-driven instantiation from YAML.
    """
    def __init__(self, params, name="compound_loss", **kwargs):
        """
        Args:
            loss_configs (list): List of dicts specifying each loss type, weight, and params.
            name (str): Optional name for the loss.
        """
        super().__init__(name=name)
        self.losses = []
        self.weights = []
        for entry in params:
            if isinstance(entry, collections.abc.Mapping):
                loss_type = entry.get("type")
                weight = entry.get("weight", 1.0)
                params = entry.get("params", {})
                if loss_type == "si_sdr":
                    loss_fn = SISDRLoss(**params)
                elif loss_type == "mrl_mse":
                    loss_fn = MultiResolutionSTFTLossFromSTFT(**params)
                elif loss_type == "log_mse":
                    loss_fn = LogFramewiseMSE(**params)
                elif loss_type == "mse":
                    loss_fn = FramewiseMSE(**params)
                elif loss_type == "mae":
                    loss_fn = FramewiseMAE(**params)
                elif loss_type == "compressed_mse":
                    loss_fn = CompressedMSE(**params)
                else:
                    raise ValueError(f"Unknown loss type: {loss_type}")
                self.losses.append(loss_fn)
                self.weights.append(weight)

        if len(self.losses) != len(self.weights):
            raise ValueError("Length of losses and weights must be the same.")

    def call(self, y_true, y_pred, **kwargs):
        """
        Compute the weighted sum of the individual losses.

        Args:
            y_true: Ground truth tensor
            y_pred: Predicted tensor

        Returns:
            Scalar tensor representing the combined loss
        """
        total_loss = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(y_true, y_pred, **kwargs)
        return total_loss

# --- Factory for CompoundLoss from config ---
def build_compound_loss(config):
    """
    Factory to build CompoundLoss from config dict (YAML parsed).
    Args:
        config (dict): Should contain 'losses' key with list of loss configs.
    Returns:
        CompoundLoss instance
    """
    losses = config.get("losses", [])
    return CompoundLoss(losses)
