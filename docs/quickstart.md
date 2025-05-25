# Quickstart Guide

## Install SoundKit

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

## Requirements

- Python >= **3.10**

**Optional (for EVB demo support):**

- [Arm GNU Toolchain](https://developer.arm.com/downloads/-/gnu-rm) >= **12.2**
- [Segger J-Link](https://www.segger.com/downloads/jlink/) >= **7.92**

---

## Setup Virtual Environment

To isolate your project dependencies, it's recommended to use a virtual environment:

```bash
python -m venv .venv         # Create a virtual environment
source .venv/bin/activate    # Activate it (use `.venv\Scripts\activate` on Windows)
```

---

## Install Python Dependencies

Install all necessary packages, including editable SoundKit installation:

```bash
pip install -e .
```

> This setup is ideal for development and enables instant updates to source code without reinstalling.

---

## Use SoundKit with CLI

SoundKit provides a unified command-line interface to manage tasks such as data preprocessing, model training, evaluation, exporting, and running demos.

!!! note "Syntax"

    **Usage:**

    ```bash
    soundkit --task [TASK] --mode [MODE] --config [CONFIG]
    ```

    **Arguments:**

    - `TASK` – One of: `se`, `vad`, `kws`

    - `MODE` – One of: `data`, `train`, `evaluate`, `export`, `demo`

    - `CONFIG` – Path to a YAML configuration file

!!! note "Example of Speech Enhancement (SE) Task"

    === "Data"

        Download and prepare training and validation data by generating TFRecords from raw audio corpora.

        ```bash
        soundkit -t se -m data -c configs/se.yaml
        ```

    === "Train"

        Train a speech enhancement model using the specified configuration and dataset.

        ```bash
        soundkit -t se -m train -c configs/se.yaml
        ```

        To monitor training progress in real-time, open a new terminal and launch TensorBoard:

        ```bash
        soundkit -t se -m train --tensorboard -c configs/se.yaml
        ```

        This will open TensorBoard with logs from the training run. Visit [http://localhost:6006](http://localhost:6006) in your browser to view metrics and visualizations.

    === "Evaluate"

        Evaluate the trained model on a test set and compute metrics such as SI-SDR, STOI, PESQ, or DNSMOS.

        ```bash
        soundkit -t se -m evaluate -c configs/se.yaml
        ```

    === "Export"

        Convert the trained model into formats suitable for embedded or web deployment (e.g., TFLite, C arrays).

        ```bash
        soundkit -t se -m export -c configs/se.yaml
        ```

    === "Demo"

         Deploy on [Ambiq's family of ultra-low power SoCs](https://ambiq.com/soc/) and Run real-time inference
        ```bash
        soundkit -t se -m demo -c configs/se.yaml
        ```
