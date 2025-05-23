# Speech Enhancement (SE)

This section describes the Speech Enhancement (SE) task within **SoundKit**, focusing on denoising speech signals for real-time and embedded applications. The SE module supports data preparation, model training, evaluation, export, and live demo — including edge deployment and browser-based inference.

---

## Features

- Noise suppression for clean speech recovery  
- Real-time frame-by-frame inference  
- Modular support for CRNN and UNet architectures  
- Export for embedded deployment (TFLite, CMSIS, etc.)  
- Web demo via WebUSB  

---

## Install SoundKit

Follow the instructions in the [SoundKit installation guide](../install.md) to set up your environment.

---

## SE Task Modes

Use the `soundkit` CLI to run various modes of the SE pipeline. Below is a summary of each supported mode:

!!! note "SE Task Mode Selection"

    === "Data"
        Prepare the training, validation, and test data by generating TFRecords from raw audio corpora.

        ```bash
        soundkit -t se -m data -c your_config.yaml
        ```


        | Parameter               | Description                                                                 |
        |-------------------------|-----------------------------------------------------------------------------|
        | `path_tfrecord`         | Path to store generated TFRecords                                           |
        | `tfrecord_datalist_name`| CSV files listing TFRecords for training and testing                        |
        | `num_samples_per_noise` | Number of samples per noise file (per split)                                |
        | `force_download`        | If true, redownloads datasets                                               |
        | `reverb_prob`           | Probability of applying reverb augmentation                                 |
        | `num_processes`         | Number of parallel workers                                                  |
        | `corpora`               | List of datasets used for data generation                                   |
        | `snr_dbs`               | List of SNR levels (in dB)                                                  |
        | `target_length_in_secs` | Length of generated clips in seconds                                        |
        | `min_amp`, `max_amp`    | Amplitude range applied to randomly scale the synthesized audio                                  |
        | `signal`                | Dictionary defining STFT and signal preprocessing parameters                |


    === "Train"
        Train the speech enhancement model using the specified configuration and dataset.

        ```bash
        soundkit -t se -m train -c your_config.yaml
        ```
        
        | Parameter            | Description                                                                 |
        |----------------------|-----------------------------------------------------------------------------|
        | `initial_lr`         | Initial learning rate for the optimizer. Cosine schedule is utilized here.                                     |
        | `batchsize`          | Mini-batch size used during training                                        |
        | `epochs`             | Total number of training epochs                                             |
        | `warmup_epochs`      | Number of epochs for linear learning rate warm-up                          |
        | `epoch_loaded`       | `random`, `latest` or an integer. Epoch number to resume training from       |
        | `loss_function`      | `mse` or `compressed_mse`. Type and parameters for the loss function (e.g., `compressed_mse`)          |
        | `loss_function.params.exp` | Exponent used in compressed magnitude loss                            |
        | `loss_function.params.eps` | Small epsilon to avoid division by zero                               |
        | `path.models_trained`| Directory to save trained model checkpoints                                |
        | `path.tensorboard`   | Directory to save TensorBoard logs                                          |
        | `num_lookahead`      | An integer 0 - 5. Number of lookahead frames used in training (e.g., > 0 for non-causal models)       |
        | `feature.type`       | `mel`, `logpsec` or `hybrid`. Type of feature extraction                            |
        | `feature.bins`       | Number of feature bins (e.g., mel bins or FFT bins)                         |
        | `standardization`    | If `true`, apply mean-variance normalization to features                    |
        | `model.config_dir`   | Directory containing architecture YAML files                                |
        | `model.config_file`  | Specific YAML file describing model architecture (e.g., `config_crnn.yaml`)|


    === "Evaluate"
        Evaluate the model on a test set and compute metrics such as SI-SDR, STOI, PESQ, or DNSMOS.

        ```bash
        soundkit -t se -m evaluate -c your_config.yaml
        ```

        | Parameter               | Description                                                                 |
        |--------------------------|-----------------------------------------------------------------------------|
        | `epoch_loaded`           | Epoch number of the trained model to evaluate                              |
        | `data.from`              | Source of evaluation data (`raw` for direct WAV input, or `tfrecord`)       |
        | `data.dir`               | Directory containing raw audio WAV files (used if `data.from` is `raw`)     |
        | `data.files`             | List of filenames (WAV) to evaluate, relative to `data.dir`                 |


    === "Export"

        
        Convert the trained model into formats suitable for embedded or web deployment (e.g., TFLite, C arrays).

        ```bash
        soundkit -t se -m export -c your_config.yaml
        ```

        | Parameter         | Description                                                         |
        |-------------------|---------------------------------------------------------------------|
        | `epoch_loaded`    | Epoch number of the trained model to export                         |
        | `tflite_dir`      | Directory where the TFLite model (or exported C files) will be saved|

    === "Demo"
        Run real-time inference either on:
        - A connected embedded development board (EVB), or  
        - In-browser using WebUSB (Chrome-based browsers only)

        ```bash
        soundkit -t se -m demo -c your_config.yaml
        ```

        | Parameter         | Description                                                              |
        |-------------------|--------------------------------------------------------------------------|
        | `epoch_loaded`    | Epoch number of the trained model used for real-time inference           |
        | `tflite_dir`      | Directory containing the exported TFLite model                           |
        | `evb_dir`         | Path to embedded board (EVB) project files for deployment and testing    |

---

## Example Config

Below is an example `se.yaml` configuration file for training a CRNN-based speech enhancement model using 16kHz audio, 480-sample frames, and no lookahead.

<details>
<summary><strong>se.yaml (SoundKit Config)</strong></summary>

```yaml
name: your_model_name
project: se
job_dir: ./soundkit/tasks/se
debug: false

data:
  path_tfrecord: ./soundkit/tasks/se/tfrecords
  tfrecord_datalist_name:
    train: train_tfrecord.csv
    test: test_tfrecord.csv
  num_samples_per_noise:
    train: 1000
    test: 250
  force_download: false
  reverb_prob: 0.2
  num_processes: 8
  corpora:
    - {name: train-clean-360, type: train, path: wavs/LibriSpeech/train-clean-360}
    - {name: train-clean-100, type: train, path: wavs/LibriSpeech/train-clean-100}
    - {name: dev-clean, type: dev, path: wavs/LibriSpeech/dev-clean}
    - {name: test-clean, type: test, path: wavs/LibriSpeech/test-clean}
    - {name: musan, type: noise}
    - {name: wham_noise, type: noise}
    - {name: rirs_noises, type: reverb}
  snr_dbs: [-6, -3, 0, 3, 6, 9, 12, 15, 30]
  target_length_in_secs: 5
  min_amp: 0.01
  max_amp: 0.95

  signal:
    sampling_rate: 16000
    frame_size: 480
    hop_size: 160
    fft_size: 512
    dc_removal: true

train:
  initial_lr: 4e-4
  batchsize: 32
  epochs: 150
  warmup_epochs: 5
  epoch_loaded: random
  loss_function:
    type: compressed_mse
    params:
      exp: 0.6
      eps: 1e-8
  path:
    models_trained: ./soundkit/tasks/se/models_trained
    tensorboard: ./soundkit/tasks/se/tensorboard
  num_lookahead: 0
  feature:
    type: mel
    bins: 72
  standardization: true

  model:
    config_dir: ./soundkit/models/arch_configs
    config_file: config_crnn.yaml

evaluate:
  epoch_loaded: 129
  data:
    from: raw
    dir: "./wavs/se/test_wavs"
    files: [keyboard_steak.wav, i_like_steak.wav]

export:
  epoch_loaded: 149
  tflite_dir: ./soundkit/tasks/se/tflite

demo:
  epoch_loaded: 50
  tflite_dir: ./soundkit/tasks/se/tflite
  evb_dir: ./soundkit/tasks/se/evb
```