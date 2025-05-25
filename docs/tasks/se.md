# Speech Enhancement (SE)

The **Speech Enhancement (SE)** module in **SoundKit** enables denoising of speech signals for real-time and embedded applications. It is designed for both research and deployment, supporting:

* ✅ Data preparation
* ✅ Model training
* ✅ Evaluation
* ✅ Model export
* ✅ Real-time inference (demo)

This module is optimized for deployment on [Ambiq's family of ultra-low power SoCs](https://ambiq.com/soc/), enabling efficient and low-latency speech enhancement on edge devices.

📘 **Try it now:** Explore the [SE Tutorial Notebook](../../notebooks/SoundKit_SE_Tutorial.ipynb) for a hands-on walkthrough.

---


---

## Features

- Noise suppression for clean speech recovery  
- Real-time frame-by-frame inference  
- Modular support for [CRNN](../models/crnn.md) and [UNet](../models/unet.md) architectures  
- Export for embedded deployment (TFLite, CMSIS, etc.)  
- Demo on [Ambiq's family of ultra-low power SoCs](https://ambiq.com/soc/) via WebUSB  

---

## Install SoundKit

Follow the instructions in the [QuickStart](../quickstart.md) to set up your environment.

---

## SE Task Modes


??? example "`config_se.yaml`"
    ```yaml
    name: crnn150_lookahead0
    project: se
    job_dir: ./soundkit/tasks/se

    data:
      path_tfrecord: ${job_dir}/tfrecords
      tfrecord_datalist_name:
        train: train_tfrecord.csv 
        val: val_tfrecord.csv
      num_samples_per_noise:
        train: 1000
        val: 250
      ...
    ```


Use the `soundkit` CLI to run various modes of the SE pipeline. Below is a summary of each supported mode:

!!! note "SE Task Mode Selection"

    === "Data"
        Download and prepare the training and validation data by generating TFRecords from raw audio corpora.

        ```bash
        soundkit -t se -m data -c configs/se.yaml
        ```
    === "Train"
        Train the speech enhancement model using the specified configuration and dataset.

        ```bash
        soundkit -t se -m train -c configs/se.yaml
        ```

        To monitor training progress in real-time, open a new terminal and launch TensorBoard:

        ```bash
        soundkit -t se -m train --tensorboard -c configs/se.yaml
        ```
        This will open TensorBoard with logs from the specified training run. Visit http://localhost:6006 in your browser to view metrics and visualizations.


    === "Evaluate"
        Evaluate the model on a test set and compute metrics such as SI-SDR, STOI, PESQ, or DNSMOS.

        ```bash
        soundkit -t se -m evaluate -c configs/se.yaml
        ```

    === "Export"

        
        Convert the trained model into formats suitable for embedded or web deployment (e.g., TFLite, C arrays).

        ```bash
        soundkit -t se -m export -c configs/se.yaml
        ```

    === "Demo"
        Run real-time inference either on:
        - A connected embedded development board (EVB), or  
        - In-browser using WebUSB (Chrome-based browsers only)

        ```bash
        soundkit -t se -m demo -c configs/se.yaml
        ```
---


