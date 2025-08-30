from pydantic import BaseModel
from typing import Optional, List
import tensorflow as tf

class UNetParams(BaseModel):
    name: str = 'unet'
    batchsize: int = 8
    time_steps: int = 1
    dim_feat: int = 257
    dim_out: int = 257
    kernel_size_time_en: int = 2
    kernel_size_time_de: int = 2
    kernel_size_freq: int = 3
    num_chs: List[int] = [1, 16, 16, 16, 16]
    separable: bool = False
    activation: str = 'relu'
    unroll_rnn: bool = False
    normalization_layer: str | None = None
    dropout: float = 0.0
    output_activation: str = 'sigmoid'

    middle_net: Optional[str] = 'tcn' # 'lstm' or 'tcn' or None

    rnn_res: bool = False
    skip_connection_type: str = 'concat'

    kernel_size_tcn: int = 3
    dilations: List[int] = [1, 2, 4, 8, 16, 32]


def get_unet_info(
        num_chs: list = [1, 2, 4, 8, 16],
        kernel_size_freq: int = 3,
        dim_feat: int = 257,
    ):
    """
    Print the information of the UNet:
    return:
        - freq_bins:
            Number of frequency bins in each stage. It includes
            the input (dim_feat)
            Formula: bins_update = (bins - kernel_size_freq) // 2 + 1
        - pad_freq_bins: 
            List of the number of padding frequency bins in each stage
    """
    stages = len(num_chs) - 1
    freq_bins = [dim_feat]
    num_bin = dim_feat
    pad_freq_bins = []
    for _ in range(stages):
        pad_freq_bins += [(num_bin - kernel_size_freq) % 2]
        num_bin = (num_bin - kernel_size_freq) // 2 + 1
        freq_bins += [num_bin]
    return freq_bins, pad_freq_bins

class SliceLayer(tf.keras.layers.Layer):
    """ Slice layer"""
    def __init__(
            self,
            kernel_size_time=3,
            **kwargs):
        super(SliceLayer, self).__init__(**kwargs)
        self.kernel_size_time = kernel_size_time

    def call(self, inputs):
        """ Forward pass"""
        outputs=inputs[:,(self.kernel_size_time-1):-(self.kernel_size_time-1),:,:] # remove the last (kernel_size_time-1) frames

        return outputs
