# 📁 Data Preparation

This page explains how to prepare training, validation, and test datasets for **Speech Enhancement (SE)** using the `soundkit` CLI.

The dataset preparation process mixes clean speech with noise (and optional reverb), applies SNR scaling and amplitude augmentation, and saves the synthesized examples into TFRecords for training.

---

## 🔧 Run `data` Mode

```bash
soundkit -t se -m data -c your_config.yaml
```

---

## 🧾 Data Parameters

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

## 📦 Corpora Definition

Corpora include clean speech, noise, and reverb impulse responses.

```yaml
corpora:
  - {name: train-clean-360, type: train, path: wavs/LibriSpeech/train-clean-360}
  - {name: dev-clean, type: val, path: wavs/LibriSpeech/dev-clean}
  - {name: ESC-50-master, type: noise}
  - {name: rirs_noises, type: reverb}
```

| `type` Value | Meaning |
|--------------|---------|
| `train`, `val`, `test` | Clean speech data |
| `train-val` | One corpus used for both train and val (e.g., THCHS-30) |
| `noise` | Noise-only datasets |
| `reverb` | Room impulse responses for reverb simulation |

> 🧠 Nested paths for `train-val` (e.g. THCHS-30) can include a structure like:
> ```yaml
> - {name: thchs30, type: train-val, path: {train: path/to/train, dev: path/to/dev}}
> ```

---

## 🎚 Signal Preprocessing

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

## 🧪 Output

Running the `data` mode will generate:

- TFRecord files (e.g., `train-00001.tfrecord`) at `./soundkit/tasks/se/tfrecords`
- CSV index files (`train_tfrecord.csv`, `val_tfrecord.csv`) referencing TFRecord shards

These are required by the `train`, `evaluate`, and `export` steps.
