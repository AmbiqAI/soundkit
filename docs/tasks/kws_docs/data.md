
# Data Preparation for Keyword Spotting (KWS)

The data preparation step in **SoundKit** for KWS involves generating training and validation datasets by mixing keyword utterances with background noise and non-keyword speech at controlled SNR levels.

This process supports reproducible and diverse datasets suitable for training robust keyword detection models.

---

## Overview

The `data` mode prepares TFRecord datasets by:
- Combining keywords with garbage speech and environmental noise
- Controlling signal-to-noise ratio (SNR)
- Configuring reverberation and amplitude thresholds
- Supporting frame extension for keyword detection alignment

Run with:

```bash
soundkit -t kws -m data -c configs/kws/kws.yaml
```

---

## Configuration Parameters (`kws.yaml`)

Here are the key settings for the `data` section:

```yaml
data:
  path_tfrecord: ${job_dir}/tfrecords
  tfrecord_datalist_name:
    train: train_tfrecord.csv 
    val: val_tfrecord.csv
  num_samples_per_noise:
    train: 757
    val: 177
  force_download: false
  reverb_prob: 0.0
  num_processes: 8
  corpora:
    - {name: train-galaxy, type: speech, split: train}
    - {name: val-galaxy, type: speech, split: val}
    - {name: vad_train-clean-100, type: garbage, split: train}
    - {name: vad_train-clean-360, type: garbage, split: train}
    - {name: vad_dev-clean, type: garbage, split: val}
    - {name: vad_thchs30, type: garbage, split: train-val}
    - {name: ESC-50-master, type: noise, split: train-val}
    - {name: FSD50K, type: noise, split: train-val}
    - {name: musan, type: noise, split: train-val}
    - {name: wham_noise, type: noise, split: train-val}
    - {name: rirs_noises, type: reverb, split: train-val}
  snr_dbs: [-12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 30]
  target_length_in_secs: 15
  min_amp: 0.01
  max_amp: 0.95

  signal:
    sampling_rate: 16000
    dc_removal: true
  target_frames_extension: 30
  debug: false
```
## Data Parameters
| Parameter                 | Description                                              | Example/Values                                       |
| ------------------------- | -------------------------------------------------------- | ---------------------------------------------------- |
| `path_tfrecord`           | Output directory for TFRecords                           | `${job_dir}/tfrecords`                               |
| `tfrecord_datalist_name`  | Filenames listing TFRecord entries                       | `train: train_tfrecord.csv`, `val: val_tfrecord.csv` |
| `num_samples_per_noise`   | Number of samples per noise type                         | `train: 757`, `val: 177`                             |
| `force_download`          | Force data re-download if `true`                         | `false`                                              |
| `reverb_prob`             | Probability of adding reverberation                      | `0.0`                                                |
| `num_processes`           | Number of parallel processes for data prep               | `8`                                                  |
| `corpora`                 | List of keyword, garbage, noise, and reverb data sources | See detailed list below                              |
| `snr_dbs`                 | Range of SNRs used for augmentation                      | `[-12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 30]`          |
| `target_length_in_secs`   | Duration of each audio sample in seconds                 | `15`                                                 |
| `min_amp`                 | Minimum amplitude threshold for clipping                 | `0.01`                                               |
| `max_amp`                 | Maximum amplitude threshold for clipping                 | `0.95`                                               |
| `signal.sampling_rate`    | Audio sampling rate in Hz                                | `16000`                                              |
| `signal.dc_removal`       | Apply DC offset removal if `true`                        | `true`                                               |
| `target_frames_extension` | Number of frames to extend ground truth target span      | `30`                                                 |

---

## Notes

- **target_frames_extension**: Extends the ground truth window to capture the full keyword span.
- **corpora**: Keywords, garbage speech, and noise sources should be explicitly defined.
- **snr_dbs**: Defines the range of SNR values for augmenting robustness.

See the [QuickStart Guide](../../quickstart.md) if you're running this step for the first time.
