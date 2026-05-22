""" 
Utility functions to download
and manage TensorFlow model checkpoints.
"""
import os
import re
from typing import Tuple, Union,  List, Dict, Any
import json
from omegaconf import OmegaConf
import tensorflow as tf
from soundkit.defines import SKTaskParams
from soundkit.models import ModelFactory, ModelParamFactory

def load_model_checkpoint(
    model: tf.keras.Model,
    epoch_loaded: Union[str, int],
    checkpoint_root: str,
    criterion_epoch: str = 'val_loss',
) -> Tuple[int, int]:
    """
    Load model weights from checkpoint based on the specified epoch.

    Args:
        model (tf.keras.Model): The model to load weights into.
        epoch_loaded (str | int): One of:
            - "random": Skip loading, start from scratch.
            - "latest": Load the most recent checkpoint.
            - int: Load checkpoint from specific epoch number.
        checkpoint_root (str): Model folder containing the 'checkpoints/' directory.

    Returns:
        Tuple[int, int]: 
            - epoch_loaded: Actual epoch number loaded (-1 if 'random').
            - epoch1_loaded: Next epoch to start training from (epoch_loaded + 1).
    """
    if epoch_loaded == 'random':
        return -1, 0

    checkpoint_dir = f'{checkpoint_root}/checkpoints'

    if epoch_loaded == 'latest':
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

        model.load_weights(latest_checkpoint)

        match = re.search(r'_ep(\d+)', latest_checkpoint)
        if not match:
            raise ValueError(
                f"Cannot extract epoch number from checkpoint name: {latest_checkpoint}")

        epoch_loaded = int(match.group(1))

    elif epoch_loaded == 'best':

        # Load your JSON log
        with open(f"{checkpoint_root}/train_log.json", "r") as f:
            logs = json.load(f)
        # Find the entry with the lowest test_loss
        # ✅ Remove None entries
        logs = [entry for entry in logs if entry is not None]

        # ✅ Find entry with lowest val_loss
        if criterion_epoch == 'val_loss':
            best_epoch_entry = min(logs, key=lambda x: x[criterion_epoch])
        elif criterion_epoch == 'val_acc':
            best_epoch_entry = max(logs, key=lambda x: x[criterion_epoch])
        epoch_loaded = best_epoch_entry['epoch']
        checkpoint_path = f'{checkpoint_dir}/model_checkpoint_ep{epoch_loaded}'

        model.load_weights(checkpoint_path)
        print(f"Loaded best model from epoch {epoch_loaded} amoung {criterion_epoch}")

    else:
        # Load a specific epoch number
        epoch_loaded = int(epoch_loaded)
        checkpoint_path = f'{checkpoint_dir}/model_checkpoint_ep{epoch_loaded}'
        model.load_weights(checkpoint_path)

    return epoch_loaded, epoch_loaded + 1

def save_train_log(train_log: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save training log to a JSON file.

    Args:
        train_log: A list of dictionaries (one per epoch).
        filepath: Where to save the JSON log.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(train_log, f, indent=2)


def load_train_log(
        filepath: str,
        epochs: int) -> List[Dict[str, Any]]:
    """
    Load training log from a JSON file.

    Args:
        filepath: Path to the log file.

    Returns:
        train_log: List of logged metrics per epoch.
    """
    if not os.path.exists(filepath):
        return [None] * epochs  # Return empty log for all epochs if file does not exist

    with open(filepath, "r") as f:
        train_log = json.load(f)
    return train_log


def get_model_config(params: SKTaskParams) -> Dict[str, Any]:
    """Load the model arch config and merge task-level overrides."""
    model_config = "model"
    config_path = f"{params.train[model_config]['config_dir']}/{params.train[model_config]['config_file']}"

    config_dict = OmegaConf.load(config_path)

    if "override" in params.train[model_config] and params.train[model_config]["override"] is not None:
        override_cfg = OmegaConf.create(params.train[model_config]["override"])
        config_dict = OmegaConf.merge(config_dict, override_cfg)

    OmegaConf.resolve(config_dict)
    return OmegaConf.to_container(config_dict, resolve=True)


def build_model(
        params: SKTaskParams,
        batchsize: int = 32,
        dim_feat: int = 72,
        time_steps: int = 1,
        export: bool = False,
        summary: bool = True) -> Tuple[tf.keras.Model, int]:

    """Download model weights from a remote server.

    Args:
        params (SKTaskParams): Task parameters
    """
    print(f"Downloading model weights for {params.name} to {params.job_dir}")

    # 1.1. Build the model

    # Load from YAML file

    config_dict = get_model_config(params)

    model_name=config_dict['name']

    if export:
        config_dict['unroll_rnn'] = True
        

    Params_Cls = ModelParamFactory.get(model_name)

    params_net = Params_Cls(
        dim_feat=dim_feat,
        batchsize=batchsize,
        time_steps=time_steps,
        **config_dict)

    model = ModelFactory.get(
        model_name,
        params=params_net)

    if hasattr(model, 'complex'):
        if model.complex:
            inputs_x = tf.keras.Input(
                shape=[time_steps, dim_feat, 2], batch_size=batchsize, dtype=tf.float32, name="x_input")
        else:
            inputs_x = tf.keras.Input(
                shape=[time_steps, dim_feat], batch_size=batchsize, dtype=tf.float32, name="x_input")
    else:
        inputs_x = tf.keras.Input(
            shape=[time_steps, dim_feat], batch_size=batchsize, dtype=tf.float32, name="x_input")

    model(inputs_x)
    if summary:
        model.summary()

    return model
