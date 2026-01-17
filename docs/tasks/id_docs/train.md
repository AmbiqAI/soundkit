# 🏋️‍♂️ Training (Speaker Verification - ID)

This page explains how to train a **Speaker Verification (ID)** model using the `soundkit` CLI. The training setup includes speaker-labeled TFRecords, configurable embeddings, feature extraction, loss functions, and scheduling.

---

## 🚀 Run `train` Mode

```bash
soundkit -t id -m train -c configs/id/id.yaml
```

This command launches training using your settings in `id.yaml`.

To monitor training in real time using TensorBoard, open a second terminal:

```bash
soundkit -t id -m train -c configs/id/id.yaml --tensorboard
```

Then visit [http://localhost:6006](http://localhost:6006) in your browser.

---

## 🧾 Training Parameters

| Parameter | Description |
|-----------|-------------|
| `initial_lr` | Initial learning rate (uses cosine decay by default) |
| `lr schedule` | Learning rate scheduling strategy (e.g., cosine, step, or custom). |
| `batchsize` | Batch size for speaker embedding training |
| `epochs` | Number of total training epochs |
| `warmup_epochs` | Warm-up period to ramp up learning rate |
| `epoch_loaded` | Resume strategy: `random`, `latest`, `best`, or specific epoch |
| `loss_function.type` | `cross_entropy` or `focal_loss` |
| `path.checkpoint_dir` | Path to save model checkpoints |
| `path.tensorboard_dir` | Path to store TensorBoard logs |
| `num_lookahead` | Number of lookahead frames (0 = causal inference) |
| `feature` | Feature extraction settings (e.g., type, bins, frame size, hop size, FFT size). See below for YAML format. |
| `standardization` | Enables per-feature mean-variance normalization for input features. |
| `model` | Model architecture configuration (directory, config file, and overrides). See below for YAML format. |
| `reset_states_every_batch` | If `true`, resets model states after each batch (useful for stateful RNNs). |

---

## 🎛 Feature Extraction

```yaml
feature:
  frame_size: 480
  hop_size: 160
  fft_size: 512
  type: mel
  bins: 40
```

| Parameter | Description |
|-----------|-------------|
| `type` | Feature type: `mel`, `logpspec`, or `hybrid` |
| `bins` | Number of mel or FFT bins |
| `frame_size` | STFT window size in samples |
| `hop_size` | Frame shift in samples |
| `fft_size` | Length of FFT used in STFT |

These must align with your data preparation settings.

---

## 🔄 Standardization

```yaml
standardization: true
```

Enables per-feature mean-variance normalization for better training stability.

---

## 🧠 Model Configuration

Model configuration is specified as:

```yaml
model:
  config_dir: ./soundkit/models/arch_configs
  config_file: config_crnn_id.yaml
```

Example CRNN-style architecture for speaker ID:

```yaml
override:
  units: 100
  len_time: 6
  stride_time: 1
  dropout_rate_input: 0.1
  dropout_rate: 0.25
```

You can edit `config_crnn_id.yaml` or switch to a different architecture by changing `config_file`.

For adding your own models, refer to [BYOM](../../models/byom.md).

---

## 📦 Output

Once training completes:

- Model checkpoints are saved in `checkpoint_dir`
- Logs and metrics are written to `tensorboard_dir`
- The best model can be used for evaluation and export using the `epoch_loaded: best` flag

---

To visualize training progress again:

```bash
soundkit -t id -m train --tensorboard -c configs/id/id.yaml
```

Let SoundKit help you build robust, low-power speaker ID models for embedded and on-device voice authentication.
