# 📁 **Data Preparation (Voice Activity Detection - VAD)**

This page outlines how to prepare training and validation datasets for **Voice Activity Detection (VAD)** using the `soundkit` CLI. This step synthesizes examples by combining clean speech and noise (optionally applying reverb), sets SNR levels, and performs amplitude scaling before saving the outputs to TFRecords.

---

## 🔧 **Run `data` Mode**

```bash
soundkit -t vad -m data -c vad/vad.yaml
```

---

## 🧾 **Data Parameters**

| Parameter | Description |
|-----------|-------------|
| `path_tfrecord` | Output path for storing generated TFRecords. Set to `${job_dir}/tfrecords` |
| `tfrecord_datalist_name` | CSV file listing TFRecord shards for training and validation |
| `num_samples_per_noise` | Number of synthesized samples per noise clip for each split (`train`, `val`) |
| `force_download` | Forces re-download of corpora if set to `true` |
| `reverb_prob` | Probability of applying reverb via impulse response recordings |
| `num_processes` | Number of worker processes used during data synthesis |
| `snr_dbs` | List of signal-to-noise ratios (in dB) used for mixing clean speech and noise |
| `target_length_in_secs` | Duration of each generated audio clip in seconds |
| `min_amp`, `max_amp` | Amplitude range to randomly scale each sample |
| `debug` | Enables verbose logging for debugging if set to `true` |

---

## 📦 **Corpora Configuration**

Corpora define which datasets are mixed during data generation. Each entry in the `corpora` field includes:

- `name`: Dataset name (must match a registered loader)
- `type`: One of `speech`, `noise`, or `reverb`
- `split`: Dataset subset (`train`, `val`, or `train-val`)

```yaml
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
```

To register your own dataset, refer to the [BYOD](../../datasets/byod.md) documentation.

---

## 🎚 **Signal Preprocessing**

The `signal` section defines how the waveform is prepared before feature extraction:

```yaml
signal:
  sampling_rate: 16000
  dc_removal: true
```

| Parameter | Description |
|-----------|-------------|
| `sampling_rate` | Sampling rate (Hz) for all audio |
| `dc_removal` | Removes DC bias before performing STFT if set to `true` |

---

## 🧪 **Output**

After running the `data` mode:

- TFRecord files (e.g., `train-00001.tfrecord`) are generated in `${job_dir}/tfrecords`
- CSV files (`train_tfrecord.csv`, `val_tfrecord.csv`) list the TFRecord files
- These outputs are consumed during the training and evaluation phases
