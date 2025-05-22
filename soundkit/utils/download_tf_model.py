import os
import re
import yaml
from typing import Tuple, Union,  List, Dict, Any
import tensorflow as tf
import json
from ..defines import SKTaskParams
from ..models import ModelFactory, ModelParamFactory

def load_model_checkpoint(
    model: tf.keras.Model,
    epoch_loaded: Union[str, int],
    model_dir: str
) -> Tuple[int, int]:
    """
    Load model weights from checkpoint based on the specified epoch.

    Args:
        model (tf.keras.Model): The model to load weights into.
        epoch_loaded (str | int): One of:
            - "random": Skip loading, start from scratch.
            - "latest": Load the most recent checkpoint.
            - int: Load checkpoint from specific epoch number.
        model_dir (str): Model folder containing the 'checkpoints/' directory.

    Returns:
        Tuple[int, int]: 
            - epoch_loaded: Actual epoch number loaded (-1 if 'random').
            - epoch1_loaded: Next epoch to start training from (epoch_loaded + 1).
    """
    if epoch_loaded == 'random':
        return -1, 0

    checkpoint_dir = f'{model_dir}/checkpoints'

    if epoch_loaded == 'latest':
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
        
        model.load_weights(latest_checkpoint)

        match = re.search(r'_ep(\d+)', latest_checkpoint)
        if not match:
            raise ValueError(f"Cannot extract epoch number from checkpoint name: {latest_checkpoint}")
        
        epoch_loaded = int(match.group(1))

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
    with open(filepath, "w") as f:
        json.dump(train_log, f, indent=2)


def load_train_log(filepath: str) -> List[Dict[str, Any]]:
    """
    Load training log from a JSON file.

    Args:
        filepath: Path to the log file.

    Returns:
        train_log: List of logged metrics per epoch.
    """
    if not os.path.exists(filepath):
        return []

    with open(filepath, "r") as f:
        train_log = json.load(f)
    return train_log


def build_model(
        params: SKTaskParams,
        batchsize: int = 32,
        dim_feat: int = 72,
        export: bool=False) -> Tuple[tf.keras.Model, int]:
    """Download model weights from a remote server.

    Args:
        params (SKTaskParams): Task parameters
    """
    print(f"Downloading model weights for {params.name} to {params.job_dir}")


    # 1.1. Build the model

    # Load from YAML file

    config_path = f"{params.train['model']['config_dir']}/{params.train['model']['config_file']}"
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    model_name=config_dict['name']
    if export:
        time_steps = 1
    else:
        time_steps = params.data['target_length_in_secs'] * 100    
    
    
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
    
    uv = tf.ones(shape=(batchsize, time_steps, dim_feat), dtype=tf.float32)
    model(uv)
    
    # for v in model.trainable_variables:
    #     print(f"{v.name}: {v.numpy().shape}")
    
    
    # import pickle
    # with open("soundkit/tasks/se/arrays.pkl", "rb") as f:
    #     arr_list_loaded = pickle.load(f)
    # for i, v in enumerate(  model.trainable_variables):
    #     print(f"{i}: {v.name}: {v.numpy().shape}")
    #     v.assign(arr_list_loaded[i])
    # import pdb; pdb.set_trace()    
    # import pdb; pdb.set_trace()
    # 1.2. Load model weights from checkpoint
    # _, epoch_loaded_1 = load_model_checkpoint(
    #     model, params_train['epoch_loaded'], model_dir)
    return model