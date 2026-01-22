
# Training for Keyword Spotting (KWS)

The training step in **soundKIT** for KWS involves optimizing a CRNN-based model to detect keywords in audio streams under varied and noisy conditions.

This module supports focal or cross-entropy loss, dynamic SNR data mixing, and TensorBoard visualization.

---

## Running Training

Use the following command to start training:

```bash
soundkit -t kws -m train -c configs/kws/kws.yaml
```

To monitor training with TensorBoard:

```bash
soundkit -t kws -m train --tensorboard -c configs/kws/kws.yaml
```

---

## Configuration (`train` section of `kws.yaml`)

```yaml
train:
  initial_lr: 4e-4
  batchsize: 128
  epochs: 150
  warmup_epochs: 5
  epoch_loaded: random
  loss_function:
    type: focal
    params: {gamma: 3.0, alpha: 0.75}

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
      units: 64
      dropout_rate_input: 0.1
      dropout_rate: 0.3
      stride_time: 1
      len_time: 6

  reset_every_batch: false
```
## Training Parameters
| Parameter                    | Description                                              | Value / Example |
| ---------------------------- | -------------------------------------------------------- | --------------- |
| `initial_lr`                 | Initial learning rate                                    | `4e-4`          |
| `lr_schedule`                | Learning rate scheduling strategy (e.g., cosine, step)   | `cosine`        |
| `batchsize`                  | Batch size used during training                          | `128`           |
| `epochs`                     | Number of training epochs                                | `150`           |
| `warmup_epochs`              | Number of warmup epochs for LR scheduler                 | `5`             |
| `epoch_loaded`               | Specifies how training starts (e.g., `random`, `latest`) | `random`        |
| `loss_function.type`         | Loss function type                                       | `focal`         |
| `loss_function.params.gamma` | Focal loss focusing parameter                            | `3.0`           |
| `loss_function.params.alpha` | Focal loss balancing factor                              | `0.75`          |
| `path`                      | Dictionary of output paths for checkpoints and logs      | See below       |
| `num_lookahead`              | Number of lookahead frames (0 = causal inference)        | `0`             |
| `feature`                    | Feature extraction settings (type, bins, frame size, etc.)| See below       |
| `standardization`            | Enables per-feature mean-variance normalization          | `true`          |
| `model`                      | Model architecture configuration (directory, config, overrides) | See below |
| `reset_states_every_batch`   | If `true`, resets model states after each batch          | `false`         |

## Paths
| Parameter              | Description                                        | Value / Example                                     |
| ---------------------- | -------------------------------------------------- | --------------------------------------------------- |
| `path.full_name`       | Dynamic name format based on model hyperparameters | `${name}_loss-${train.feature.type}_...`            |
| `path.checkpoint_dir`  | Checkpoint directory path                          | `${job_dir}/models_trained/${train.path.full_name}` |
| `path.tensorboard_dir` | TensorBoard logs path                              | `${job_dir}/tensorboard/${train.path.full_name}`    |

## Feature Extraction
| Parameter            | Description              | Value / Example |
| -------------------- | ------------------------ | --------------- |
| `feature.frame_size` | Frame size in samples    | `480`           |
| `feature.hop_size`   | Hop size in samples      | `160`           |
| `feature.fft_size`   | FFT size                 | `512`           |
| `feature.type`       | Feature type             | `logpspec`      |
| `feature.bins`       | Number of frequency bins | `257`           |

---

## Notes

- **Loss Function**: Use `focal` for imbalanced detection or `cross_entropy` for standard classification.
- **Features**: `logpspec` (log power spectrum) is used for frame-wise input features.
- **Model Override**: CRNN hyperparameters like dropout and temporal stride can be customized.

Refer to the [Data Preparation](./data.md) section before starting training.
