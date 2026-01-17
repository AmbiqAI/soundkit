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

* $x_i$ and $y_i$ are real and > 0 (e.g., clean and noisy amplitudes of a spectrogram)  
* $\epsilon$ is a small number to avoid singularities during gradient descent  
* $0 < r \leq 1$  
* $r < 1$ applies compression to reduce the influence of high-magnitude values

This loss is particularly useful in speech enhancement or audio applications where human perception is non-linear.

---

# Cross Entropy Loss

The **Cross Entropy Loss** is a standard loss function for classification tasks. It measures the dissimilarity between the predicted probability distribution and the true distribution:

$$
\text{CrossEntropy}(\mathbf{p}, \mathbf{y}) = - \sum_i y_i \log(p_i)
$$

where:

* $\mathbf{p}$ is the vector of predicted probabilities (output of softmax)  
* $\mathbf{y}$ is the one-hot encoded ground truth vector  
* The sum runs over all classes $i$

Cross Entropy is minimized when the predicted probability for the correct class is 1 and all others are 0. It strongly penalizes confident but incorrect predictions.

# Focal Loss

The **Focal Loss** is designed to address class imbalance in classification tasks by down-weighting easy examples and focusing training on hard negatives:

$$
\text{Focal Loss}(p_t) = -\alpha_t\,(1 - p_t)^\gamma\,\log(p_t)
$$

where:

* $p_t$ is the model's estimated probability for the true class  
* $\alpha_t$ is a balancing factor to address class imbalance (optional)  
* $\gamma \geq 0$ is the focusing parameter: higher values of $\gamma$ focus more on hard, misclassified examples

Focal Loss reduces the impact of easily classified samples ($p_t \rightarrow 1$), enabling the model to concentrate more on difficult or misclassified examples. This mechanism is especially useful in highly imbalanced classification problems.

**Example: Keyword Spotting (KWS)**:
The dataset may contain very few positive samples (i.e., actual keywords) compared to a large number of negative (non-keyword) samples. A naive neural network could achieve high accuracy by simply predicting the negative class most of the time, due to the class imbalance.

A common approach to mitigating this is to apply a class weighting factor (such as $\alpha_t$) to give more importance to the minority class. However, Focal Loss provides an automatic mechanism by leveraging the model’s own confidence:

- When a prediction is correct and confident (e.g., $p_t = 0.99$), the modulating factor $(1 - p_t)^\gamma$, spontaneously, becomes very small (e.g., $(1 - 0.99)^2 = 0.0001$ when $\gamma = 2$), greatly reducing the loss contribution from this easy example.

- In contrast, incorrect or uncertain predictions (low $p_t$) contribute significantly more to the loss, guiding the model to focus its learning effort where it is needed most.

This dynamic scaling of the loss helps prevent the model from being overwhelmed by the majority class and improves performance on underrepresented classes without manual tuning.

---