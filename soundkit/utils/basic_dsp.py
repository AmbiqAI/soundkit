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
