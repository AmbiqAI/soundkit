# CRNN Model Configuration

This is a simple CRNN (Convolutional Recurrent Neural Network) configuration supporting the following layer types:

* `conv2d`
* `lstm`
* `fc` (fully connected)
* `dropout`
* `batchnorm`
* `layernorm`

## Example Configuration (`config_crnn.yaml`)

```yaml
name: crnn_simple
layer_configs:
  - type: conv2d
    filters: 64
    kernel_size: [3, 3]
    strides: [1, 1]
    activation: relu

  - type: batchnorm
    momentum: 0.99
    epsilon: 1e-3

  - type: dropout
    rate: 0.2

  - type: lstm
    units: 128

  - type: layernorm
    epsilon: 1e-5

  - type: fc
    units: 64
    activation: relu

  - type: dropout
    rate: 0.3

  - type: fc
    units: 1
    activation: linear
```

## Layer Descriptions

* `conv2d`: 2D convolutional layer for spatial feature extraction
* `batchnorm`: Batch normalization layer

  * `momentum`: Decay rate for moving average (e.g. `0.99`)
  * `epsilon`: Small constant to avoid division by zero (e.g. `1e-3`)
* `dropout`: Regularization layer to prevent overfitting

  * `rate`: Fraction of input units to drop (e.g. `0.2`)
* `lstm`: Long Short-Term Memory layer for sequential modeling

  * `units`: Number of hidden units
* `layernorm`: Layer normalization across features

  * `epsilon`: Small constant for numerical stability (e.g. `1e-5`)
* `fc`: Fully connected (dense) layer

  * `units`: Number of neurons
  * `activation`: Activation function (e.g. `relu`, `sigmoid`, `linear`)

> Only the above layers and parameters are allowed. Ensure your configuration adheres to this format for compatibility.
