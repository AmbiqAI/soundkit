"""VAD task model export function."""
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from soundkit.defines import SKTaskParams

from soundkit.utils.calculate_feat_stats import load_feat_stats
from soundkit.utils.tflite_convert import (
    tflite_convert,
    warp_tf_model
)
from soundkit.utils.download_tf_model import (
     build_model,
     load_model_checkpoint
)
from soundkit.utils.tf_copy_model import copy_model_weights
from soundkit.utils.calculate_feat_stats import feat_stats_estimator
from soundkit.utils.TFLiteAudioModel import TFLiteAudioModel
from soundkit.utils.basic_dsp import DCRemover
from soundkit.utils.feature_utils import FeatureExtractor
from soundkit.utils.np_feature_utils import FeatureExtractor_np
from soundkit.datasets import SKDatasetFactory
from soundkit.utils.audio import audio_read
from soundkit.utils.calculate_feat_stats import mean_varinace_norm
from .datasets import create_dataset
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)

def export(params: SKTaskParams):
    """
    Export the VAD model to TFLite format.
    """
    params_export = params.export
    hop_size = params.train['feature']['hop_size']
    tflite_filename_src = f"{params.name}_{params.export['dtype']}.tflite"
    tflite_path_src = Path(params.export['tflite_dir']) / tflite_filename_src
    checkpoint_dir = f"{params.train['path']['checkpoint_dir']}"

    batchsize_train = params.train['batchsize']
    batchsize = 1
    feat_extractor = FeatureExtractor(
        params=params,
        )
    dim_feat = feat_extractor.dim_feat

    # 1.1. Build the model
    # Load from YAML file

    if params.train['truncate_time'] is not None:
        time_steps = int(
            params.train['truncate_time'] \
            * params.data.signal.sampling_rate \
            // params.train.feature.hop_size)
    else:
        time_steps = int(
            params.data['target_length_in_secs'] \
            * params.data.signal.sampling_rate \
            //  params.train.feature.hop_size)

    model_train = build_model(
        params,
        batchsize=batchsize_train,
        dim_feat=dim_feat,
        time_steps=time_steps)

    load_model_checkpoint(
        model_train, params_export['epoch_loaded'], checkpoint_dir)

    model = build_model(
        params,
        batchsize=batchsize,
        dim_feat=dim_feat,
        time_steps=1,
        export=True)

    copy_model_weights(model_dst=model, model_src=model_train)
    if params.train.feature.type in ('spec','erb_complex'):
        is_complex = True
    else:
        is_complex = False
    model_wrap = warp_tf_model(
        model,
        time_steps=1,
        dim_feat=dim_feat,
        is_complex=is_complex)

    path_tflite=f'{params.export["tflite_dir"]}/{params.name}_{params.export["dtype"]}.tflite'
    
    # Prepare calibration data for quantization if needed
    if params.export.calibration_samples is not None:
        tfrecord_list = {
            'train': 
                Path(params.data['path_tfrecord']) / params.data['tfrecord_datalist_name']['train'],
            'val': 
                Path(params.data['path_tfrecord']) / params.data['tfrecord_datalist_name']['val'],
        }

        ds_train, _ = create_dataset(
            tfrecord_list['train'],
            batchsize=params.train['batchsize'],
            is_shuffle=True,
            hop_size=params.train['feature']['hop_size'],
        )
        # 1. Flatten the dataset and extract the exact number of required calibration samples.
        # We discard additional labels/metadata using a map, keeping only the raw features.
        ds_collected = ds_train.unbatch() \
                            .take(params.export.calibration_samples) \
                            .map(lambda x, *args: x) \
                            .batch(params.export.calibration_samples)

        # 2. Materialize the dataset into a single Tensor.
        # next(iter()) is used here to efficiently pull the first (and only) batch 
        # into memory as a TensorFlow constant.
        audio_sn_tf = next(iter(ds_collected))

        # 3. Compute features using the GPU-accelerated extractor.
        # We process the entire calibration set as one batch and convert to a 
        # NumPy array only at the final step for downstream compatibility.
        data_calibration = feat_extractor(audio_sn_tf)[0].numpy()
        # 4. Compute feature statistics for standardization
        if params.train['standardization']:
            stats = feat_stats_estimator(
                ds_train,
                batchsize_train,
                folder_nn=checkpoint_dir,
                feat_extractor=feat_extractor,)
        else:
            stats = {
                'nMean_feat': tf.zeros([dim_feat], dtype=tf.float32),
                'nInvStd': tf.ones([dim_feat], dtype=tf.float32),
            }
        # for complex features-handling, split real and imaginary parts
        if np.iscomplexobj(data_calibration):
            data_calibration = np.stack(
                [np.real(data_calibration), np.imag(data_calibration)],
                axis=-1)
        if params.train['standardization']:
            # Standardize features
            data_calibration = mean_varinace_norm(data_calibration, stats['nMean_feat'], stats['nInvStd'])
        else:
            # No standardization, use raw features
            data_calibration = data_calibration

    else:
        data_calibration = None
    tflite_fp16_model = tflite_convert(
        model_wrap,
        dtype=params.export.dtype,
        path_tflite=path_tflite,
        qbits=params.export.qbit_input,
        data_calibration=data_calibration,
    )
    print(f"Exported TFLite model saved at: {tflite_path_src}")

    vad_model = build_vad_tflite(params, str(tflite_path_src))

    if params.export.eval:
        corpus={"name": "vad_dev-clean", "type": "speech", "split": "val"}
        loader = SKDatasetFactory.get(corpus['name'])
        samples = loader(corpus)
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

def build_vad_tflite(
        params: SKTaskParams,
        tflite_path_src: str):
    """Export VAD task model with given parameters.

    Args:
        params (SKTaskParams): Task parameters
        tflite_path_src: str
    """
    hop_size = params.train['feature']['hop_size']
    checkpoint_dir = f"{params.train['path']['checkpoint_dir']}"

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


    if params.train.standardization:
        stats = load_feat_stats(checkpoint_dir, 'stats.pkl')
    else:
        stats = None

    model_tflite = TFLiteAudioModel(
        interpreter_path=str(tflite_path_src),
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
