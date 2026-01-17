# 🏋️‍♂️ Training (Voice Activity Detection - VAD)

This page describes how to train a Voice Activity Detection (VAD) model using the `soundkit` CLI. You can configure model architecture, feature extraction, loss function, and learning schedule in your YAML config.

---

## 🚀 Run `train` Mode

```bash
soundkit -t vad -m train -c configs/vad/vad.yaml
```

This starts training using settings from `vad.yaml`, including TFRecord input, features, and model parameters.

To monitor training live, open a second terminal and run:

```bash
soundkit -t vad -m train --tensorboard -c configs/vad/vad.yaml
```

This starts TensorBoard. Visit **http://localhost:6006** to view real-time metrics and logs.

---

## 🧾 Training Parameters

| Parameter | Description |
|-----------|-------------|
| `initial_lr` | Initial learning rate (uses cosine decay) |
| `batchsize` | Batch size for training |
| `epochs` | Total number of training epochs |
| `warmup_epochs` | Warm-up period for learning rate ramp-up |
| `epoch_loaded` | Resume strategy: `random`, `latest`, `best`, or a specific epoch number |
| `loss_function.type` | Loss function (e.g., `cross_entropy`) |
| `path.checkpoint_dir` | Where to save model checkpoints |
| `path.tensorboard_dir` | Where to save TensorBoard logs |
| `num_lookahead` | Lookahead frames used (0 for causal) |

---

## 🎛 Feature Extraction

```yaml
feature:
  frame_size: 480
  hop_size: 160
  fft_size: 512
  type: logpspec
  bins: 257
```

| Parameter | Description |
|-----------|-------------|
| `type` | Feature type: `logpspec`, `mel`, or `hybrid` |
| `bins` | FFT or mel bin count |
| `frame_size` | STFT window size in samples |
| `hop_size` | Frame hop size |
| `fft_size` | FFT length used in STFT |

These settings must match those used for TFRecord generation.

---

## 🔄 Standardization

```yaml
standardization: true
```

Enable mean-variance normalization during training for improved convergence.

---

## 🧠 Model Configuration

Specify the model architecture via:

```yaml
model:
  config_dir: ./soundkit/models/arch_configs
  config_file: config_crnn_vad.yaml
```

A sample CRNN VAD configuration:

```yaml
name: crnn
units: 22
len_time: 6
dropout_rate_input: 0.1
dropout_rate: 0.2
stride_time: 1
layer_configs:
  - type: dropout
    rate: ${dropout_rate_input}

  - type: fc
    units: ${units}
    activation: relu

  - type: dropout
    rate: ${dropout_rate}

  - type: conv2d
    filters: ${units}
    kernel_size: ["${len_time}", "${units}"]
    strides: ["${stride_time}", 1]
    activation: relu

  - type: dropout
    rate: ${dropout_rate}

  - type: lstm
    units: ${units}

  - type: dropout
    rate: ${dropout_rate}

  - type: fc
    units: ${units}
    activation: relu

  - type: dropout
    rate: ${dropout_rate}

  - type: fc
    units: ${units}
    activation: relu

  - type: dropout
    rate: ${dropout_rate}

  - type: fc
    units: 2
    activation: linear
```

You can switch between CRNN and other architectures by modifying the config. For custom models, see [BYOM](../../models/byom.md).

---

## 📦 Output

After training completes:

- Checkpoints saved to `checkpoint_dir`
- Training logs in `tensorboard_dir`
- Model ready for evaluation and export using the same `name` and `epoch_loaded`

To visualize metrics:

```bash
soundkit -m train --tensorboard -c configs/vad/vad.yaml
```
