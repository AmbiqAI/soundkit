# Speech Enhancement (SE)

The [Speech Enhancement (SE)](./se_docs/introduction.md) module in **soundKIT** enables denoising of speech signals for real-time and embedded applications. It is designed for both research and deployment, supporting:

* ✅ [Data preparation](./se_docs/data.md)
* ✅ [Model training](./se_docs/train.md)
* ✅ [Evaluation](./se_docs/evaluate.md)
* ✅ [Model export](./se_docs/export.md)
* ✅ [Real-time inference (demo)](./se_docs/demo.md)

This module is optimized for deployment on [Ambiq's family of ultra-low power SoCs](https://ambiq.com/soc/), enabling efficient and low-latency speech enhancement on edge devices.

📘 **Try it now:** Explore the [SE Tutorial Notebook](/notebooks/SoundKit_SE_Tutorial/) for a hands-on walkthrough.

---


---

## Features

- Noise suppression for clean speech recovery
- Real-time frame-by-frame inference
- Modular support for [CRNN](../models/crnn.md) and [UNet](../models/unet.md) architectures
- Export for embedded deployment (TFLite, CMSIS, etc.)
- Demo on [Ambiq's family of ultra-low power SoCs](https://ambiq.com/soc/) via WebUSB

---

## Install soundKIT

Follow the instructions in the [QuickStart](../quickstart.md) to set up your environment.

---

## SE Task Modes
The `soundkit` CLI provides multiple modes for running the SE task. All modes are configured through a YAML file (e.g., `se.yaml`). Below is a breakdown of the configuration structure and CLI commands.

??? example "`se.yaml`"
    ```yaml
    name: unet_experiment
    project: se
    job_dir: ./soundkit/tasks/s

    data:
      path_tfrecord: ${job_dir}/tfrecords
      tfrecord_datalist_name: # list of saved tfrecords
        train: train_tfrecord.csv
        val: val_tfrecord.csv
      num_samples_per_noise:
        train: 1000
        val: 250
      force_download: false
      reverb_prob: 0.5
      num_processes: 8
      corpora:
        - {name: train-clean-360, type: speech, split: train}
        - {name: train-clean-100, type: speech, split: train}
        - {name: dev-clean, type: speech, split: val,}
        - {name: thchs30, type: speech, split: train-val}
        - {name: ESC-50-master, type: noise, split: train-val}
        - {name: FSD50K, type: noise, split: train-val}
        - {name: musan, type: noise, split: train-val}
        - {name: wham_noise, type: noise, split: train-val}
        - {name: rirs_noises, type: reverb, split: train-val}
      snr_dbs: [-6, -3, 0, 3, 6, 9, 12, 15, 30] # mixture of signal-to-noise ratios
      target_length_in_secs: 5
      min_amp: 0.03
      max_amp: 0.95

      signal:
        sampling_rate: 16000
        dc_removal: true
      debug: false

    train:
      initial_lr: 4e-4
      batchsize: 32
      epochs: 150
      warmup_epochs: 5
      epoch_loaded: random
      loss_function: {
        type: compressed_mse,
        params: {exp: 0.6, eps: 1e-8}
        }
      path:
        full_name: ${name}_unit64_la${train.num_lookahead}_dropout0.2_${train.feature.type}_feat
        model_dir:       ${job_dir}/models_trained/${train.path.full_name}
        tensorboard_dir: ${job_dir}/tensorboard/${train.path.full_name}
      num_lookahead: 2

      feature:
        frame_size: 480
        hop_size: 160
        fft_size: 512
        type: logpspec
        bins: 257
        # type: hybrid
        # bins_fft: 100
        # n_mels: 72

      standardization: true

      model:
        config_dir: ./soundkit/models/arch_configs
        config_file: config_unet.yaml

      debug: false

    evaluate:
      epoch_loaded: best

      data:
        dir: "./wavs/se/test_wavs"
        files: [keyboard_steak.wav, i_like_steak.wav, steak_hairdryer.wav]
        # # dir: ./wavs/LibriSpeech/test-clean
        # # files:
        result_folder: ${job_dir}/test_results/${train.path.full_name}

    export:
      epoch_loaded: best
      tflite_dir: ${job_dir}/tflite

    demo:
      platform: pc
      epoch_loaded: best
      tflite_dir: ${job_dir}/tflite
      evb_dir: ${job_dir}/evb
      pre_gain: 1


    ```

!!! note "SE Task Mode Selection"

    === "Data"
        Download and prepare the training and validation data by generating TFRecords from raw audio corpora.

        ```bash
        soundkit -t se -m data -c configs/se/se.yaml
        ```
        See [Data](./se_docs/data.md) in detail.
    === "Train"
        Train the speech enhancement model using the specified configuration and dataset.

        ```bash
        soundkit -t se -m train -c configs/se/se.yaml
        ```

        To monitor training progress in real-time, open a new terminal and launch TensorBoard:

        ```bash
        soundkit -t se -m train --tensorboard -c configs/se/se.yaml
        ```
        This will open TensorBoard with logs from the specified training run. Visit http://localhost:6006 in your browser to view metrics and visualizations.

        See [Train](./se_docs/train.md) in detail.

    === "Evaluate"
        Evaluate the model on a test set and compute metrics such as SI-SDR, STOI, PESQ, or DNSMOS.

        ```bash
        soundkit -t se -m evaluate -c configs/se/se.yaml
        ```
        See See [Evaluate](./se_docs/evaluate.md) in detail.

    === "Export"


        Convert the trained model into formats suitable for embedded or web deployment (e.g., TFLite, C arrays).

        ```bash
        soundkit -t se -m export -c configs/se/se.yaml
        ```
        See [Export](./se_docs/export.md) in detail.
    === "Demo"
        Run real-time inference either on:
        - On your PC
        - A connected embedded development board (EVB) via WebUSB
        We suggest to use PC first for fast testing.
        ```bash
        soundkit -t se -m demo -c configs/se/se.yaml demo.platform=pc
        # or
        soundkit -t se -m demo -c configs/se/se.yaml demo.platform=evb
        ```
        See [Demo](./se_docs/demo.md) in detail.
---
