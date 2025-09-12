import numpy as np
import tensorflow as tf
from soundkit.utils.np_feature_utils import FeatureExtractor_np

class TFLiteAudioModel:
    """ A general-purpose wrapper for TensorFlow Lite audio models.
    This class handles preprocessing (feature extraction and normalization),
    TFLite model inference, and basic post-processing.
    It supports both quantized and float models, allowing for flexible input/output types.
    It can be used for various audio tasks such as speech recognition, classification, etc.
    """
    def __init__(
        self,
        interpreter: tf.lite.Interpreter,
        dtype: str = "float32",
        *args, **kwargs
    ):
        """
        A general-purpose wrapper for TensorFlow Lite audio models.

        This class handles preprocessing (feature extraction and normalization),
        TFLite model inference, and basic post-processing.

        Args:
            interpreter (tf.lite.Interpreter): Loaded TFLite interpreter.
            hop_size (int): Output granularity in samples (used for reshaping).
            stats (dict | None): Optional normalization stats with keys 'nMean_feat' and 'nInvStd'.
            dtype (str): Input/output precision, e.g., 'float32', 'int8', or 'int16'.
        """
        super().__init__(*args, **kwargs)
        
        self.interpreter = interpreter
        self.dtype = dtype

        self.input_details = interpreter.get_input_details()[0]
        self.output_details = interpreter.get_output_details()[0]
        
    def __call__(
            self,
            input_tensor: np.ndarray) -> np.ndarray:
        """
        Runs the model on input audio.

        Args:
            x (np.ndarray): 1D or multi-dimensional audio input array.

        Returns:
            np.ndarray: Model output reshaped to match input layout.
        """

        # Handle quantized inputs
        if self.dtype in ("int8", "int16"):
            scale, zero_point = self.input_details["quantization"]
            input_tensor = input_tensor / scale + zero_point
            input_tensor = input_tensor.astype(np.int8 if self.dtype == "int8" else np.int16)
        else:
            input_tensor = input_tensor.astype(np.float32)
        self.interpreter.set_tensor(self.input_details['index'], input_tensor)


        # Inference
        self.interpreter.invoke()
        output_tensor = self.interpreter.get_tensor(self.output_details['index'])

        # Dequantize if needed
        if self.dtype in ("int8", "int16"):
            scale, zero_point = self.output_details["quantization"]
            output = (output_tensor - zero_point).astype(np.float32) * scale
        else:
            output = output_tensor

        return output

    def reset(self):
        """Reset the TFLite model state."""
        # self.interpreter.reset_all_variables() # doesn't work
        self.interpreter.allocate_tensors()  # Reallocate tensors to reset state
