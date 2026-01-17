# UNet Model Configuration

This document describes the configuration options for a simple UNet-style model used in SoundKit.

## Example Configuration (`config_unet.yaml`)

```yaml
name: unet
kernel_size_time: 2
num_chs: [1, 64, 64, 64, 64]
separable: true
activation: relu
unroll_rnn: false
normalization_layer:
dropout: 0.0
```

## Parameter Descriptions

* `name`: Identifier for the model (e.g. `unet`)

* `kernel_size_time`: Temporal kernel size used in 1D or separable convolutions

* `num_chs`: List of channel sizes for each level of the encoder/decoder

  * First value typically corresponds to input channels
  * Last value is the number of channels after the final convolution

* `separable`: If `true`, use depthwise separable convolutions for efficiency

* `activation`: Activation function used after convolutions (e.g. `relu`, `tanh`)

* `unroll_rnn`: Reserved for architectures combining convolution and recurrent layers; unused in standard UNet

* `normalization_layer`: Optional normalization layer to use (`batchnorm`, `layernorm`, or `None`). Leave blank or unset for no normalization

* `dropout`: Dropout rate applied after layers (0.0 means no dropout)

> This configuration defines the encoder-decoder topology and layer behavior. The actual implementation may expand on this with skip connections and upsampling strategies.

---
