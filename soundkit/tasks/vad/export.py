"""VAD task model export function."""
import numpy as np
import tensorflow as tf
from soundkit.utils.tflite_convert import tflite_convert, warp_tf_model
from soundkit.defines import SKTaskParams
from soundkit.utils.download_tf_model import build_model, load_model_checkpoint
from soundkit.utils.tf_copy_model import copy_model_weights
from soundkit.utils.calculate_feat_stats import load_feat_stats
from soundkit.utils.TFLiteAudioModel import TFLiteAudioModel
from soundkit.utils.basic_dsp import DCRemover
from soundkit.utils.np_feature_utils import FeatureExtractor_np
from soundkit.datasets import SKDatasetFactory
from soundkit.utils.audio import audio_read

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

    if params.train.feature.type=='hybrid':
        mel_bins = params.train.feature.n_mels
    else:
        if "bins" in params.train.feature:
            mel_bins = params.train.feature.bins
        else:
            mel_bins = params.train.feature.frame_size

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
        model_train, params.demo['epoch_loaded'], checkpoint_dir)

    model = build_model(
        params,
        batchsize=batchsize,
        dim_feat=dim_feat,
        time_steps=1,
        export=True)

    copy_model_weights(model_dst=model, model_src=model_train)

    model_wrap = warp_tf_model(
        model,
        time_steps=1,
        dim_feat=dim_feat)

    tflite_fp16_model = tflite_convert(
        model_wrap,
        dtype=params.export.dtype,
        path_tflite=f'{params.export["tflite_dir"]}/{params.name}_{params.export.dtype}.tflite',)

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
                ):
            self.model_tflite = model_tflite
            self.stats = stats
            self.feat_extractor = feat_extractor
            self.is_inference = 1

            self.counts_vad_trigger = 0
            if params.data.signal.dc_removal:
                self.dc_remover = DCRemover()

        def reset(self):
            """Reset the model state."""

            self.counts_vad_trigger = 0
            self.feat_extractor.reset()
            if params.data.signal.dc_removal:
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

            outputs = self.model_tflite(features)  # Run inference

            outputs = outputs.flatten()
            if 1:
                if outputs[0] < outputs[1]:
                    outputs = np.ones(hop_size, dtype=np.float32)*0.95
                    self.counts_vad_trigger += 1
                else:
                    outputs = np.zeros(hop_size, dtype=np.float32)
                    self.counts_vad_trigger = 0
            else:
                tot = np.exp(outputs, out=outputs)  # Apply softmax
                out = tot / np.sum(tot)  # Normalize
                outputs = np.ones(hop_size, dtype=np.float32)*out[1]

            if self.counts_vad_trigger ==180:
                # self.reset()
                self.counts_vad_trigger = 0
            outputs = outputs.reshape(shape)
            return outputs

    vad_model = VADModel(
        model_tflite=model_tflite,
        feat_extractor=feat_extractor,
        stats=stats,
    )
    return vad_model

def export(params: SKTaskParams):
    """
    Export the VAD model to TFLite format.
    """
    hop_size = params.train['feature']['hop_size']
    vad_model = build_vad_tflite(params)

    corpus={"name": "vad_dev-clean", "type": "speech", "split": "val"}
    loader = SKDatasetFactory.get(corpus['name'])
    samples = loader(corpus)


    if params.export.eval:
        cmat_acc = tf.zeros((2,2), dtype=tf.int64)
        for idx_sample , sample in enumerate(samples):
            
            print(f"\rProcessing {idx_sample+1}/{len(samples)}", end='')
            vad_model.reset()
            wav, label = sample
            
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
        print("\nConfusion matrix (rows: true, cols: pred):")
        cmat_acc = cmat_acc / tf.reduce_sum(cmat_acc, axis=-1, keepdims=True)
        print(cmat_acc.numpy())
