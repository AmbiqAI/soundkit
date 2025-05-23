from pathlib import Path
import numpy as np
import tensorflow as tf
import torch
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
        
from .datasets import create_dataset
from ...defines import SKTaskParams
from ...utils.download_tf_model import build_model, load_model_checkpoint
from ...utils.feature_utils import FeatureExtractor
from ...utils.calculate_feat_stats import feat_stats_estimator
from ...utils.lookaheadBuffer import LookaheadBuffer
from ...utils.tf_stft import tf_istft
from ...utils.tf_complex_utils import polar_to_complex
from ...utils.tf_copy_model import copy_model_weights

def evaluate(params: SKTaskParams):
    """Evaluate SE task model with given parameters.

    Args:
        params (HKTaskParams): Task parameters
    """
    print(f"Evaluating SE model with params: {params} and more")

    params_evaluate = params.evaluate
    num_lookahead = params.train['num_lookahead']
    signal = params.data['signal']
    model_dir = f"{params.train['path']['models_trained']}/{params.name}"

    batchsize_train = params.train['batchsize']
    batchsize = 32
    dim_feat = params.train['feature']['bins']

    # 1.1. Build the model

    # Load from YAML file
    model_train = build_model(
        params,
        batchsize=batchsize_train,
        dim_feat=dim_feat)
    load_model_checkpoint(
        model_train, params_evaluate['epoch_loaded'], model_dir)

    model = build_model(
        params,
        batchsize=batchsize,
        dim_feat=dim_feat)

    copy_model_weights(
        model_dst=model,
        model_src=model_train)

    # 2. Create the dataset
    tfrecord_list = {
        'test':  Path(params.data['path_tfrecord']) / params.data['tfrecord_datalist_name']['test'],
    }

    dataset, batches = create_dataset(
        tfrecord_list['test'],
        batchsize=batchsize,
        is_shuffle=True,
    )

    # 3. Define feature extractor
    feat_extractor = FeatureExtractor(
        params=params,
    )

    # 4. Compute feature statistics for standardization
    stats = feat_stats_estimator(
        dataset,
        batches,
        dim_feat,
        folder_nn=model_dir,
        feat_extractor=feat_extractor,)
    # Initialize left-over state buffers for streaming STFT
    states_audio_sn = tf.zeros(
        [batchsize, signal["frame_size"] - signal["hop_size"]],
        dtype=tf.float32
    )
    num_fft_bins = signal["fft_size"] // 2 + 1
    buffer_sn = LookaheadBuffer(
        num_lookahead=num_lookahead,
        feature_dim=num_fft_bins,
        batchsize=batchsize)

    np_scores = np.zeros( (1, 4), dtype=np.float64)
    for step, batch in enumerate(dataset):
        print(f"\rEvaluating (batch) {step}/{batches}, ", end='')

        audio_sn, audio_s, _ = batch

        feat_sn, spec_sn, states_audio_sn = feat_extractor(
            audio_sn, states=states_audio_sn)
        # Apply lookahead
        spec_sn_delay = buffer_sn.apply(spec_sn)

        if params.train['standardization']:
            # Standardize features

            mean_stats = stats['nMean_feat']
            inv_std_stats = stats['nInvStd']
            feat_sn_norm = (feat_sn - mean_stats) * inv_std_stats

        tfmask = model(feat_sn_norm, training=False)
        pspec_sn_delay = tf.abs(spec_sn_delay)
        phase_sn_delay = tf.math.angle(spec_sn_delay)

        pspec_en_delay = tfmask * pspec_sn_delay

        spec_en_delay = polar_to_complex(
            pspec_en_delay, phase_sn_delay)

        audio_en = tf_istft(
            spec_en_delay,
            signal['frame_size'],
            signal['hop_size'],
            signal['fft_size'])

        torch_tensor = torch.from_numpy(audio_en.numpy())
        scores = deep_noise_suppression_mean_opinion_score(
            torch_tensor,
            signal['sampling_rate'],
            False)

        np_scores += tf.reduce_sum(scores, axis=0, keepdims=True).numpy()
    np_scores /= (batches * batchsize)
    print(f"DNSMOS Score: {np_scores}[p808_mos, mos_sig, mos_bak, mos_ovr]")
  