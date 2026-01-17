# 📁 **Data Preparation (Speaker Verification - ID)**

This guide explains how to prepare training and validation datasets for **Speaker Verification (ID)** using the `soundkit` CLI. In this task, speaker-labeled utterances are augmented and organized into TFRecords suitable for training an embedding-based identity recognition model.

---

## 🔧 **Run `data` Mode**

```bash
soundkit -t id -m data -c configs/id/id.yaml
```

---

## 🧾 **Data Parameters**

| Parameter | Description |
|-----------|-------------|
| `path_tfrecord` | Output path for storing generated TFRecords. Defaults to `${job_dir}/tfrecords` |
| `tfrecord_datalist_name` | YAML files listing TFRecord shards used during training/validation (`train_tfrecord.yaml`, `val_tfrecord.yaml`) |
| `force_download` | If `true`, forces re-download of the listed datasets |
| `reverb_prob` | Probability of applying reverberation during sample generation |
| `num_processes` | Number of parallel worker processes for data synthesis |
| `corpora` | List of dataset specifications (name, type, split) used for generating speaker-labeled and noise examples. See below for YAML format. |
| `snr_dbs` | List of signal-to-noise ratios (in dB) used when mixing speech and noise |
| `target_length_in_secs` | Duration of each generated audio clip |
| `min_amp`, `max_amp` | Amplitude range for random scaling of waveform samples |
| `signal` | Dictionary of signal-level preprocessing options (e.g., sampling rate, DC removal) applied to audio before feature extraction. See below for YAML format. |
| `num_sentences` | Number of utterances sampled per speaker for ID training |
| `ppls_per_group` | Number of speakers grouped per batch for training contrastive loss models |

---

## 📦 **Corpora Configuration**

Corpora specify which datasets are used for generating speaker-labeled examples:

```yaml
corpora:
  - {name: vad_train-clean-100, type: speech, split: train}
  - {name: vad_train-clean-360, type: speech, split: train}
  - {name: vad_train-other-500, type: speech, split: train}
  - {name: vad_dev-clean, type: speech, split: val}
  - {name: ESC-50-master, type: noise, split: train-val}
  - {name: FSD50K, type: noise, split: train-val}
  - {name: musan, type: noise, split: train-val}
  - {name: wham_noise, type: noise, split: train-val}
  - {name: rirs_noises, type: reverb, split: train-val}
```

- `type: speech`: Provides speaker-labeled utterances.
- `type: noise`: Mixed with speech to create variable SNR conditions.
- `type: reverb`: Impulse responses for applying environmental reverb.

To register your own dataset, see [BYOD](../../datasets/byod.md).

---

## 🎚 **Signal Preprocessing**

Signal-level options defined in the `signal` section control waveform transformations:

```yaml
signal:
  sampling_rate: 16000
  dc_removal: true
```

| Parameter | Description |
|-----------|-------------|
| `sampling_rate` | Target sample rate for all audio |
| `dc_removal` | If `true`, removes DC bias from audio |

---

## 🧪 **Output**

Running `data` mode will generate:

- **TFRecord Files**: Saved in `${job_dir}/tfrecords`, e.g., `train-00001.tfrecord`
- **TFRecord Lists**: YAML files (`train_tfrecord.yaml`, `val_tfrecord.yaml`) that index each split
- These outputs are consumed during the training and evaluation phases of the ID pipeline

---

Ensure you’ve set up your corpora and parameters correctly before launching data preparation. For next steps, see the [Train](./train.md) guide.
