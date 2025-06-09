# Voice Activity Detection (VAD)

The [Voice Activity Detection (VAD)](./vad_docs/introduction.md) module in **SoundKit** enables robust detection of speech activity in noisy audio environments, suitable for real-time and embedded deployments. It supports:

* ✅ [Data preparation](./vad_docs/data.md)
* ✅ [Model training](./vad_docs/train.md)
* ✅ [Evaluation](./vad_docs/evaluate.md)
* ✅ [Model export](./vad_docs/export.md)
* ✅ [Real-time inference (demo)](./vad_docs/demo.md)

Optimized for edge deployment on [Ambiq's ultra-low power SoCs](https://ambiq.com/soc/), VAD ensures efficient voice detection even in constrained environments.

📘 **Try it now:** Explore the [VAD Tutorial Notebook](../../notebooks/SoundKit_VAD_Tutorial.ipynb) to get started.

---

## Features

- Frame-level voice activity prediction  
- Real-time processing for embedded or browser-based use  
- Modular architecture: use or extend CRNN-based backbones  
- TFLite and C-array export for low-power MCUs  
- Seamless EVB and WebUSB demo support

---

## Install SoundKit

See the [QuickStart Guide](../quickstart.md) to set up your environment.

---

## VAD Task Modes

The `soundkit` CLI supports multiple modes for running the VAD task. Each is configured via a YAML file (e.g., `vad.yaml`). Here's an example overview:

??? example "`vad.yaml`"
    ```yaml
    name: crnn_experiment
    project: vad
    job_dir: ./soundkit/tasks/vad

    data:
      path_tfrecord: ${job_dir}/tfrecords
      tfrecord_datalist_name:
        train: train_tfrecord.csv 
        val: val_tfrecord.csv
      num_samples_per_noise:
        train: 50000
        val: 3590
      force_download: false
      reverb_prob: 0.2
      num_processes: 8
      corpora:
        - {name: vad_train-clean-100, type: speech, split: train}
        - {name: vad_train-clean-360, type: speech, split: train}
        - {name: vad_dev-clean, type: speech, split: val}
        - {name: vad_thchs30, type: speech, split: train-val}
        - {name: ESC-50-master, type: noise, split: train-val}
        - {name: FSD50K, type: noise, split: train-val}
        - {name: musan, type: noise, split: train-val}
        - {name: wham_noise, type: noise, split: train-val}
        - {name: rirs_noises, type: reverb, split: train-val}
      snr_dbs: [-12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 30]
      target_length_in_secs: 5
      min_amp: 0.01
      max_amp: 0.95

      signal:
        sampling_rate: 16000
        dc_removal: true
      debug: false

    train:
      initial_lr: 4e-4
      batchsize: 128
      epochs: 150
      warmup_epochs: 5
      epoch_loaded: random
      loss_function: 
        type: cross_entropy
        params: {}

      path:
        full_name: ${name}_loss-${train.feature.type}_drop-${train.model.override.dropout_rate}_stridetime-${train.model.override.stride_time}_mvn-${train.standardization}_units-${train.model.override.units}_sr-${data.signal.sampling_rate}
        checkpoint_dir:  ${job_dir}/models_trained/${train.path.full_name}
        tensorboard_dir: ${job_dir}/tensorboard/${train.path.full_name}
      num_lookahead: 0

      feature:
        frame_size: 480
        hop_size: 160
        fft_size: 512
        type: logpspec
        bins: 257

      standardization: true

      model:
        config_dir: ./soundkit/models/arch_configs
        config_file: config_crnn_vad.yaml
        override:
          units: 22
          dropout_rate_input: 0.1
          dropout_rate: 0.2
          stride_time: 1
          len_time: 6

      debug: false

    evaluate:
      epoch_loaded: best
      data: 
        dir: "./wavs/vad/test_wavs"
        files: [rpc_audio_raw.wav, speech.wav, i_like_steak.wav, keyboard_steak.wav, steak_hairdryer.wav]
      result_folder: ${job_dir}/test_results/${train.path.full_name}

    export:
      epoch_loaded: best
      tflite_dir: ${job_dir}/tflite

    demo:
      platform: evb
      epoch_loaded: best
      tflite_dir: ${job_dir}/tflite
      evb_dir: ${job_dir}/evb
      pre_gain: 1
      filename: def_nn1_nnvad
      param_struct_name: params_nn1_nnvad
    ```

!!! note "VAD Task Mode Overview"

    === "Data"
        Prepare training and validation examples by mixing speech and noise with controlled SNRs and reverb.

        ```bash
        soundkit -t vad -m data -c configs/vad.yaml
        ```
        See [Data](./vad_docs/data.md) for details.

    === "Train"
        Train the VAD model with your prepared dataset and configuration.

        ```bash
        soundkit -t vad -m train -c configs/vad.yaml
        ```

        Start TensorBoard in a separate terminal:

        ```bash
        soundkit -t vad -m train --tensorboard -c configs/vad.yaml
        ```

        See [Train](./vad_docs/train.md) for guidance.

    === "Evaluate"
        Evaluate the model on real recordings to visualize predicted voice activity.

        ```bash
        soundkit -t vad -m evaluate -c configs/vad.yaml
        ```
        See [Evaluate](./vad_docs/evaluate.md) for results.

    === "Export"
        Convert model to embedded-friendly formats (TFLite, C headers) for deployment.

        ```bash
        soundkit -t vad -m export -c configs/vad.yaml
        ```
        See [Export](./vad_docs/export.md) for format descriptions.

    === "Demo"
        Test the model in real-time either using EVB hardware via WebUSB in browser:

        ```bash
        soundkit -t vad -m demo -c configs/vad.yaml
        ```
        See [Demo](./vad_docs/demo.md) to get started.
