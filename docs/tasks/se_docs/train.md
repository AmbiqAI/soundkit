
# Training

This page describes how to train a Speech Enhancement (SE) model using the `soundkit` CLI. You can customize the architecture, feature extraction, loss functions, and learning rate schedule via the configuration YAML file.

---

## Run `train` Mode

```bash
soundkit -t se -m train -c your_config.yaml
```

This command starts training using the provided configuration, including TFRecord input, feature extraction settings, and model architecture.

To monitor training progress in real-time, open a new terminal and launch TensorBoard:

```bash
soundkit -m train --tensorboard -c your_config.yaml
```

This will open TensorBoard with logs from the specified training run. Visit http://localhost:6006 in your browser to view metrics and visualizations.

---

## Training Parameters

| Parameter | Description |
|-----------|-------------|
| `initial_lr` | Initial learning rate for the optimizer. Uses cosine decay schedule |
| `batchsize` | Mini-batch size used during training |
| `epochs` | Total number of training epochs |
| `warmup_epochs` | Number of warm-up epochs for linear learning rate ramp-up |
| `epoch_loaded` | You can continue to train your model if your training procedure was interrupted for any reason. One of: <br>• `random`: start from scratch <br>• `latest`: resume from last checkpoint <br>• `best`: resume from best-performing checkpoint <br>• `<int>`: resume from a specific epoch |
| `loss_function.type` | Loss function type: [mse](../../loss.md) or [compressed_mse](../../loss.md) |
| `loss_function.params.exp` | Exponent for [compressed_mse](../../loss.md)  (e.g., 0.6) |
| `loss_function.params.eps` | Epsilon to avoid division by zero in magnitude computation (see [compressed_mse](../../loss.md))  |
| `path.checkpoint_dir` | Path to save model checkpoints |
| `path.tensorboard_dir` | Path to save TensorBoard logs |
| `num_lookahead` | Lookahead frames used during training (0 for causal models) |


---

## Feature Extraction Parameters

```yaml
feature:
  frame_size: 480
  hop_size: 160
  fft_size: 512
  type: mel
  bins: 72
```

| Parameter | Description |
|-----------|-------------|
| `type` | Feature type: `mel`, `logpsec`, or `hybrid` |
| `bins` | Number of mel bins or FFT bins |
| `frame_size` | Window size in samples |
| `hop_size` | Hop length in samples |
| `fft_size` | FFT length used for STFT |

These settings must match those used during TFRecord generation.

---

## Standardization

```yaml
standardization: true
```

If enabled, mean and variance normalization is applied to features during training.

---

## Model Configuration

Specify the architecture using a YAML file:

```yaml
model:
  config_dir: ./soundkit/models/arch_configs
  config_file: config_crnn.yaml
```

**`config_crnn.yaml`** will configure your NN:
??? example "`./soundkit/models/arch_configs/config_simple_crnn.yaml`"
    ```yaml
    name: crnn

    units: 100

    len_time: 6

    layer_configs:

    - type: dropout
        rate: 0.1

    - type: conv2d
        filters: ${units}
        kernel_size: ["${len_time}", 72]
        strides: [1, 1]
        activation: relu

    - type: lstm
        units: ${units}

    - type: fc
        units: ${units}
        activation: relu

    - type: fc
        units: ${units}
        activation: relu

    - type: fc
        units: 257
        activation: sigmoid

    ```


This allows switching between CRNN, UNet, or other registered architectures. To register your own NN architecture, see [Bring-Your-Own-Model (BYOM)](../../models/byom.md)

---

## Output

After training:

- Model checkpoints will be saved to `checkpoint_root`
- Training logs will be available in TensorBoard (`tensorboard_dir`)
- You can evaluate or export the model using the same `name` and `epoch_loaded` settings

To visualize training:

```bash
soundkit -m train --tensorboard -c your_config.yaml
```
