
---

#  Configuration Parameters (Simplified)

Understand key settings in your SoundKit YAML config for SE tasks:

## ***_ Top-Level_*****

- `name`: Name of the experiment (used in folder names)
- `project`: Task type, e.g., `se`, `kws`, `vad`
- `job_dir`: Where outputs (models, logs) are saved

## ***_ Data (`data`)_*****

- `path_tfrecord`: Where TFRecords are stored
- `corpora`: List of datasets (type: `speech`, `noise`, `reverb`)
- `snr_dbs`: List of SNR values for noise mixing (e.g., `[0, 5, 10]`)
- `target_length_in_secs`: Length of each audio clip (e.g., `5`)
- `reverb_prob`: Probability to apply reverb
- `min_amp`/`max_amp`: Controls audio amplitude range
- `signal.sampling_rate`: Sampling rate (e.g., `16000`)

## ***_ Training (`train`)_*****

- `initial_lr`: Learning rate
- `batchsize`: Batch size
- `epochs`: Total number of epochs
- `loss_function`: Type of loss and its parameters (e.g., `mrl_mse`)
- `feature`: Feature extraction settings (e.g., `type`, `frame_size`)
- `model.config_file`: Model architecture YAML (e.g., `config_crnn.yaml`)

## ***_ Evaluation (`evaluate`)_*****

- `data.dir`: Path to evaluation audio samples
- `data.files`: List of test audio files
- `result_folder`: Where results are saved

## ***_ Export (`export`)_*****

- `tflite_dir`: Exported model path (TFLite format)
- `epoch_loaded`: Which model checkpoint to export

## ***_ Demo (`demo`)_*****

- `platform`: `pc` or `evb` (Evaluation Board)
- `evb_dir`: Output directory for EVB firmware
- `filename`: Name of generated firmware
- `param_struct_name`: Struct name for exported parameters

---

# Quickstart Guide

## ***_ Install SoundKit_*****

!!! note "SE Mode Selection"

    === "From PyPI"

        ```bash
        pip install soundkit
        ```

    === "From GitHub (Development Mode)"

        ```bash
        git clone https://github.com/AmbiqAI/soundkit.git
        cd soundkit
        pip install -e .
        ```

---

## ***_ Requirements_*****

- Python  **3.10**

**Optional (for EVB demo support):**

- [Arm GNU Toolchain](https://developer.arm.com/downloads/-/gnu-rm)  **12.2**
- [Segger J-Link](https://www.segger.com/downloads/jlink/)  **7.92**

---

## ***_ Setup Virtual Environment_*****

Its best to isolate your dependencies:

```bash
python -m venv .venv         # Create virtual environment
source .venv/bin/activate    # Activate it (use `.venv\Scripts\activate` on Windows)
```

---

## ***_ Install Python Dependencies_*****

Install editable SoundKit (for dev/debug convenience):

```bash
pip install -e .
```

> Changes to the source code will immediately apply without reinstallation.

---

## ***_ Use SoundKit with CLI_*****

SoundKit provides a unified CLI for handling various ML tasks.

!!! note "Syntax"

    ```bash
    soundkit --task [TASK] --mode [MODE] --config [CONFIG]
    ```

- **TASK**  One of: `se`, `vad`, `kws`  
- **MODE**  One of: `data`, `train`, `evaluate`, `export`, `demo`  
- **CONFIG**  Path to your YAML config

---

## ***_ Example: Speech Enhancement (SE) Workflow_*****

!!! note "Common CLI Usage"

    === "Data"

        ```bash
        soundkit -t se -m data -c configs/se.yaml
        ```

    === "Train"

        ```bash
        soundkit -t se -m train -c configs/se.yaml
        ```

        Open TensorBoard in another terminal:

        ```bash
        soundkit -t se -m train --tensorboard -c configs/se.yaml
        ```

        Visit [http://localhost:6006](http://localhost:6006)

    === "Evaluate"

        ```bash
        soundkit -t se -m evaluate -c configs/se.yaml
        ```

    === "Export"

        ```bash
        soundkit -t se -m export -c configs/se.yaml
        ```

    === "Demo"

        ```bash
        soundkit -t se -m demo -c configs/se.yaml
        ```

---

## ***_ Overriding Config Values via OmegaConf_*****

SoundKit uses [OmegaConf](https://omegaconf.readthedocs.io/) for configuration management. You can override any value in the config file **directly from the CLI** using `key=value` syntax (dot notation).

## ***Example: Change platform to `evb` at runtime***

```bash
soundkit -t se -m demo -c configs/se.yaml demo.platform=evb
```

## ***Example: Override training batch size***

```bash
soundkit -t se -m train -c configs/se.yaml train.batchsize=64
```

## ***How it Works***

To support this in your script, make sure your CLI parses overrides:

```python
from omegaconf import OmegaConf
import sys

yaml_cfg = OmegaConf.load("configs/se.yaml")
cli_cfg = OmegaConf.from_dotlist(sys.argv[1:])
config = OmegaConf.merge(yaml_cfg, cli_cfg)
```

>  Do **not** prefix overrides with `--`, just use `key=value`.

---

---
## ***_ Tips_*****

- Use `OmegaConf.to_yaml(config)` to inspect the effective config at runtime.
- Keep `configs/*.yaml` clean, and override experiment-specific tweaks via CLI.
