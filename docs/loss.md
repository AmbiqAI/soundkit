# Mean Squared Error (MSE)

The **Mean Squared Error** is a commonly used loss function for regression tasks. It penalizes the squared difference between predictions and targets:

$$
\text{MSE}(\mathbf{x}, \mathbf{y}) = \sum_i (x_i - y_i)^2
$$

This loss assumes a linear scale and gives more weight to larger errors.

---

# Compressed Mean Squared Error (Compressed MSE)

The **Compressed MSE** introduces a power-law transformation to better match perceptual sensitivity or improve convergence during training. The input values are compressed by a fractional exponent before computing the squared error:

$$
\text{Compressed MSE}(\mathbf{x}, \mathbf{y}, \epsilon) = \sum_i \left((x_i+\epsilon)^r - (y_i+\epsilon)^r \right)^2
$$

where:

* $x_i$'s and $y_i$'s are real and > 0 here. For example, $x_i$ and $y_i$ are the clean and noisy amplitude of the spectrogram
* $\epsilon$ is the small number to avoid the singularity for the gradient descent
* $0 < r \leq 1$
* $r < 1$ applies compression to reduce the influence of high-magnitude values

This loss is particularly useful in speech enhancement or audio applications where human perception is non-linear.

---
