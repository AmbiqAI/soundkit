# 📁 **Data Preparation**

This page explains how to prepare training, validation, and test datasets for **Speech Enhancement (SE)** using the `soundkit` CLI.

The dataset preparation process mixes clean speech with noise (and optional reverb), applies SNR scaling and amplitude augmentation, and saves the synthesized examples into TFRecords for training.

---

## 🔧 **Run `data` Mode**

```bash
soundkit -t se -m data -c your_config.yaml
```

---

## 🧾 **Data Parameters**

| Parameter | Description |
|-----------|-------------|
| `path_tfrecord` | Output directory to store generated TFRecords. Uses `${job_dir}/tfrecords` |
| `tfrecord_datalist_name` | CSV file listing TFRecord shards for training and validation |
| `num_samples_per_noise` | Number of samples generated per noise clip for `train` and `val` splits |
| `force_download` | If `true`, forces re-download of corpora |
| `reverb_prob` | Probability of applying room reverb using impulse responses |
| `num_processes` | Number of parallel processes used for synthesis |
| `snr_dbs` | List of SNRs (in dB) for mixing clean speech with noise |
| `target_length_in_secs` | Duration of each synthesized example (in seconds) |
| `min_amp`, `max_amp` | Amplitude scaling range used to randomly scale synthesized signals |
| `debug` | If `true`, enables additional logging for debugging |

---

📦 **How Corpora Are Defined**

SoundKit uses the corpora field in YAML config files to specify the datasets to be used during training and evaluation. Each dataset is defined by:

- name: The registered name of the dataset (must match a loader function)

- type: One of speech, noise, or reverb

- split: Defines which parts of the dataset to use (train, val, or train-val)

🔧 **Default Corpora**

Below is a list of default corpora supported by SoundKit. You can find detailed descriptions in the Corpora documentation [Corpora](../../datasets/corpora.md):

```yaml
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
```

🧩 **Custom Datasets**

Want to use your own data? SoundKit makes it easy to register your own speech, noise, or reverb datasets. See the [BYOD](../../datasets/byod.md) guide for details.

---

## 🎚 **Signal Preprocessing**

The `signal` field defines how raw waveforms are prepared before feature extraction:

```yaml
signal:
  sampling_rate: 16000
  dc_removal: true
```

| Parameter | Description |
|-----------|-------------|
| `sampling_rate` | Audio sampling rate (Hz) |
| `dc_removal` | If `true`, remove DC bias before STFT |

---

## 🧪 **Output**

Running the `data` mode will generate:

- TFRecord files (e.g., `train-00001.tfrecord`) at `./soundkit/tasks/se/tfrecords`
- CSV index files (`train_tfrecord.csv`, `val_tfrecord.csv`) referencing TFRecord shards

These are required by the `train`, `evaluate`, and `export` steps.
