from pathlib import Path
import numpy as np
import tensorflow as tf
from ...utils.tflite_convert import tflite_convert, warp_tf_model
from ...defines import SKTaskParams
from ...utils.download_tf_model import build_model, load_model_checkpoint
from ...utils.tf_copy_model import copy_model_weights
from ...utils.feature_utils import FeatureExtractor
from soundkit.utils.calculate_feat_stats import load_feat_stats
from soundkit.utils.TFLiteAudioModel import TFLiteAudioModel
from soundkit.utils.basic_dsp import DCRemover
from soundkit.utils.np_feature_utils import FeatureExtractor_np

def build_vad_tflite(params: SKTaskParams):
    """Export VAD task model with given parameters.

    Args:
        params (HKTaskParams): Task parameters
    """
    hop_size = params.train['feature']['hop_size']
    sample_rate = params.data['signal']['sampling_rate']
    checkpoint_dir = f"{params.train['path']['checkpoint_dir']}"

    batchsize_train = params.train['batchsize']
    batchsize = 1
    # import pdb; pdb.set_trace()
    if params.train.feature.type=='hybrid':
        mel_bins = params.train.feature.n_mels
    elif params.train.feature.type=='time':
        mel_bins = params.train.feature.frame_size
    else:
        mel_bins = params.train.feature.bins
    feat_extractor = FeatureExtractor_np(
        feat_type=params.train.feature.type,
        frame_len=params.train.feature.frame_size,
        hop_len=params.train.feature.hop_size,
        fft_len=params.train.feature.fft_size,
        sampling_rate=params.data.signal.sampling_rate,
        mel_bins=mel_bins,
    )
    dim_feat = feat_extractor.dim_feat

    # 1.1. Build the model
    # Load from YAML file
    model_train = build_model(
        params,
        batchsize=batchsize_train,
        dim_feat=dim_feat,
        time_steps = params.data['target_length_in_secs'] * params.data.signal.sampling_rate //  params.train.feature.hop_size)

    load_model_checkpoint(
        model_train, params.export['epoch_loaded'], checkpoint_dir)

    model = build_model(
        params,
        batchsize=batchsize,
        dim_feat=dim_feat,
        time_steps=1,
        export=True)

    copy_model_weights(model_dst=model, model_src=model_train)

    if hasattr(params.export, "converter_with_reset"):
        converter_with_reset = params.export.converter_with_reset
    else:
        converter_with_reset = True

    model_wrap = warp_tf_model(
        model,
        time_steps=1,
        dim_feat=dim_feat,
        converter_with_reset=converter_with_reset,)

    if hasattr(params.export, "qbit_input"):
        qbit = params.export.qbit_input
    else:
        qbit = 8

    tflite_fp16_model = tflite_convert(
        model_wrap,
        dtype=params.export.dtype,
        qbit=qbit,
        path_tflite=f'{params.export["tflite_dir"]}/{params.name}_{params.export.dtype}.tflite',
        converter_with_reset=converter_with_reset,)

    interpreter = tf.lite.Interpreter(
        model_content=tflite_fp16_model)

    interpreter.allocate_tensors()  # Needed before execution!

    if params.train.standardization:
        stats = load_feat_stats(checkpoint_dir, 'stats.pkl')
    else:
        stats = None

    model_tflite = TFLiteAudioModel(
        interpreter=interpreter,
        dtype=params.export.dtype,
    )

    class VADModel:
        """VAD model class in tflite for processing audio input and returning VAD output."""
        def __init__(
                self,
                model_tflite: TFLiteAudioModel,
                feat_extractor: FeatureExtractor_np,
                stats: dict | None = None,
                converter_with_reset: bool = True
                ):
            self.model_tflite = model_tflite
            self.stats = stats
            self.feat_extractor = feat_extractor
            self.is_inference = 1
            self.converter_with_reset = converter_with_reset
            if converter_with_reset:
                self.reset_flag = np.array([1.0], dtype=np.float32)  # Reset flag for stateful model
            self.counts_vad_trigger = 0
            if params.data.signal.dc_removal:
                self.dc_remover = DCRemover()
            else:
                self.dc_remover = None

        def reset(self):
            """Reset the model state."""
            if self.converter_with_reset:
                self.reset_flag = np.array([1.0], dtype=np.float32)
            self.counts_vad_trigger = 0
            self.feat_extractor.reset()
            if self.dc_remover is not None:
                self.dc_remover.reset()
            self.model_tflite.reset()

        def __call__(self,
                     inputs: np.ndarray,  # input from microphone
                    ) -> np.ndarray:  # output to AudioShowClass
            """Process input audio signal and return VAD output."""
            shape=inputs.shape
            inputs=inputs.flatten()
            if params.data.signal.dc_removal:
                inputs = self.dc_remover.process(inputs)
            features,_ = self.feat_extractor(inputs)

            if self.stats is not None:
                features = (features - self.stats['nMean_feat']) * self.stats['nInvStd']

            # input to the tflite model
            features = features.reshape((1, 1, -1)) # reshape to (batch_size, time_steps, dim_feat)
            if self.converter_with_reset:
                outputs = self.model_tflite(features, reset_tensor=self.reset_flag)  # Run inference
                self.reset_flag = np.array([0.0], dtype=np.float32)  # Reset flag for next call
            else:
                outputs = self.model_tflite(features)
            
            outputs = outputs.flatten()
            if 1:
                if outputs[0] < outputs[1]:
                    outputs = np.ones(hop_size, dtype=np.float32)*0.95

                    if self.counts_vad_trigger > 0:
                        self.counts_vad_trigger += 1
                    else:
                        self.counts_vad_trigger = 1
                else:
                    outputs = np.zeros(hop_size, dtype=np.float32)
                    if self.counts_vad_trigger < 0:
                        self.counts_vad_trigger -= 1
                    else:
                        self.counts_vad_trigger = -1
            else:
                tot = np.exp(outputs, out=outputs)  # Apply softmax
                out = tot / np.sum(tot)  # Normalize
                outputs = np.ones(hop_size, dtype=np.float32)*out[1]

            # if self.counts_vad_trigger < -500:
            #     self.reset()
            # if self.counts_vad_trigger ==180:
            #     # self.reset()
            #     self.counts_vad_trigger = 0
            outputs = outputs.reshape(shape)
            return outputs

    vad_model = VADModel(
        model_tflite=model_tflite,
        feat_extractor=feat_extractor,
        stats=stats,
        converter_with_reset=converter_with_reset
    )
    return vad_model

def export(params: SKTaskParams):
    from soundkit.datasets import SKDatasetFactory
    from soundkit.utils.audio import audio_read
    hop_size = params.train['feature']['hop_size']
    vad_model = build_vad_tflite(params)


    if params.export.run_test:
        corpus={"name": "vad_test-clean", "type": "speech", "split": "val"}
        loader = SKDatasetFactory.get(corpus['name'])
        samples = loader(corpus)
        
        cmat_acc = tf.zeros((2,2), dtype=tf.int64)
        for idx_sample , sample in enumerate(samples):
            
            print(f"\rProcessing {idx_sample+1}/{len(samples)}", end='')
            # vad_model.reset()
            wav, label = sample
            # import pdb; pdb.set_trace()
            x = audio_read(wav)

            outputs = []
            for i in range(len(x)//hop_size):
                start = i*hop_size
                end = (i+1)*hop_size
                out = vad_model(x[start:end])
                outputs.append(out[0:1] > 0)

            outputs = np.concatenate(outputs)
            outputs = tf.convert_to_tensor(outputs, dtype=tf.int32)
            starts = np.array([seg['start'] for seg in label])
            ends = np.array([seg['end'] for seg in label])

            num_frames = len(x) // hop_size
            # Frame-level VAD label (vectorized scatter)
            vad = tf.zeros((num_frames,), dtype=tf.int32)

            # Create ragged index ranges
            ragged_ranges = tf.ragged.range(starts // hop_size, ends // hop_size)
            flat_indices = ragged_ranges.flat_values
            updates = tf.ones_like(flat_indices, dtype=tf.int32)

            indices = tf.expand_dims(flat_indices, axis=1)
            vad = tf.tensor_scatter_nd_update(vad, indices, updates)
            cmat = tf.math.confusion_matrix(
                labels=vad,
                predictions=outputs,
                num_classes=2)

            cmat = tf.cast(cmat, dtype=tf.int64)
            # print(cmat)
            cmat_acc += cmat
            # import pdb; pdb.set_trace()
            if 0:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6))
                plt.subplot(2, 1, 1)
                plt.plot(x, label='Audio Signal')
                plt.title('Audio Signal')
                plt.xlabel('Sample Index')
                plt.ylabel('Amplitude')

                plt.subplot(2, 1, 2)
                plt.plot(vad, label='VAD Output')
                plt.plot(outputs, label='VAD Probability', alpha=0.7)
                plt.title('VAD Output')
                plt.xlabel('Sample Index')
                plt.ylabel('Amplitude')

                plt.tight_layout()
                plt.show()
            # import pdb; pdb.set_trace()

        print("\nConfusion matrix (rows: true, cols: pred):")
        cmat_acc = cmat_acc / tf.reduce_sum(cmat_acc, axis=-1, keepdims=True)
        print(cmat_acc.numpy())
            