import numpy as np
import scipy

def dc_remove(inputs: np.float32)->np.float32:
    """remove audio dc term

    Args:
        x (float32): input

    Returns:
        float32: output
    """
    filter = scipy.signal.lfilter
    b_hpf = np.array([1.0, -1.0])
    a_hpf = np.array([1.0, -0.965267479])
    outputs = filter(b_hpf, a_hpf, inputs)
    return outputs

class DCRemover:
    def __init__(self):
        self.b_hpf = np.array([1.0, -1.0])
        self.a_hpf = np.array([1.0, -0.965267479])
        self.zi = scipy.signal.lfilter_zi(self.b_hpf, self.a_hpf)
        self.zf = self.zi * 0  # Initial state

    def process(self, inputs: np.ndarray) -> np.ndarray:
        outputs, self.zf = scipy.signal.lfilter(self.b_hpf, self.a_hpf, inputs, zi=self.zf)
        return outputs
    def reset(self):
        """Reset the internal state of the DC remover."""
        self.zf = self.zi * 0

def pre_emphasis(inputs: np.float32)->np.float32:
    """pre-emphasis

    Args:
        x (float32): input

    Returns:
        float32: output
    """
    filter = scipy.signal.lfilter
    b_hpf = np.array([1.0, -0.97])
    a_hpf = np.array([1.0])
    outputs = filter(b_hpf, a_hpf, inputs)
    return outputs

def de_emphasis(inputs: np.float32)->np.float32:
    """de-emphasis

    Args:
        x (float32): input

    Returns:
        float32: output
    """
    filter = scipy.signal.lfilter
    b_hpf = np.array([1.0])
    a_hpf = np.array([1.0,-0.97])
    outputs = filter(b_hpf, a_hpf, inputs)
    return outputs
