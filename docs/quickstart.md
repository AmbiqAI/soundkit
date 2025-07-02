# Quickstart Guide

## **Install SoundKit**

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

## **Requirements**

- Python  **3.10**

**Optional (for EVB demo support):**

- [Arm GNU Toolchain](https://developer.arm.com/downloads/-/gnu-rm)  **12.2**
- [Segger J-Link](https://www.segger.com/downloads/jlink/)  **7.92**

---

## **Setup Virtual Environment**

Its best to isolate your dependencies:

```bash
python -m venv .venv         # Create virtual environment
source .venv/bin/activate    # Activate it (use `.venv\Scripts\activate` on Windows)
```

---

## **Use SoundKit with CLI**

SoundKit provides a unified CLI for handling various ML tasks.

!!! note "Syntax"

    ```bash
    soundkit --task [TASK] --mode [MODE] --config [CONFIG]
    ```

- **TASK**  One of: `se`, `vad`, `kws`  
- **MODE**  One of: `data`, `train`, `evaluate`, `export`, `demo`  
- **CONFIG**  Path to your YAML config

---

## **Example: Speech Enhancement (SE) Workflow**

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
        soundkit -t se -m demo -c configs/se.yaml demo.platform=evb # for amibiq evb deployment
        soundkit -t se -m demo -c configs/se.yaml demo.platform=pc # for pc deployment
        
        ```

---

##  Configuration Parameters (Simplified)

Understand key settings in your SoundKit YAML config for SE tasks:

### **Top-Level**

- `name`: Name of the experiment (used in folder names)
- `project`: Task type, e.g., `se`, `kws`, `vad`
- `job_dir`: Where outputs (models, logs) are saved

### **Data (`data`)**

- `path_tfrecord`: Where TFRecords are stored
- `corpora`: List of datasets (type: `speech`, `noise`, `reverb`)
- `snr_dbs`: List of SNR values for noise mixing (e.g., `[0, 5, 10]`)
- `target_length_in_secs`: Length of each audio clip (e.g., `5`)
- `reverb_prob`: Probability to apply reverb
- `min_amp`/`max_amp`: Controls audio amplitude range
- `signal.sampling_rate`: Sampling rate (e.g., `16000`)

### **Training (`train`)**

- `initial_lr`: Learning rate
- `batchsize`: Batch size
- `epochs`: Total number of epochs
- `loss_function`: Type of loss and its parameters (e.g., `mrl_mse`)
- `feature`: Feature extraction settings (e.g., `type`, `frame_size`)
- `model.config_file`: Model architecture YAML (e.g., `config_crnn.yaml`)

### **Evaluation (`evaluate`)**

- `data.dir`: Path to evaluation audio samples
- `data.files`: List of test audio files
- `result_folder`: Where results are saved

### **Export (`export`)**

- `tflite_dir`: Exported model path (TFLite format)
- `epoch_loaded`: Which model checkpoint to export

### **Demo (`demo`)**

- `platform`: `pc` or `evb` (Evaluation Board)
- `evb_dir`: Output directory for EVB firmware
- `filename`: Name of generated firmware
- `param_struct_name`: Struct name for exported parameters

---

## **Overriding Config Values via OmegaConf**

SoundKit uses [OmegaConf](https://omegaconf.readthedocs.io/) for configuration management. You can override any value in the config file **directly from the CLI** using `key=value` syntax (dot notation).

**Example: Change platform to `evb` at runtime**

```bash
soundkit -t se -m demo -c configs/se.yaml demo.platform=evb
```

**Example: Override training batch size**

```bash
soundkit -t se -m train -c configs/se.yaml train.batchsize=64
```


## **Model Zoo**

You can directly try on already trained model on your PC.

- se:

    ```bash
    soundkit -t se -m demo -c configs/se/se.yaml demo.platform=pc # crnn
    # or
    soundkit -t se -m demo -c configs/se/se_unet.yaml demo.platform=pc # unet
    ```
- vad

    ```bash
    soundkit -t vad -m demo -c configs/vad/vad.yaml demo.platform=pc 
    ```

- kws

    ```bash
    soundkit -t kws -m demo -c configs/kws/kws.yaml demo.platform=pc
    ```

- id
    ```bash
    soundkit -t id -m demo -c configs/id/id.yaml demo.platform=pc
    ```
