"""Evaluate ID task model with given parameters."""
import os
import logging
import tensorflow as tf
from soundkit.defines import SKTaskParams
from soundkit.utils.download_tf_model import (
        build_model,
        load_model_checkpoint
    )
from soundkit.utils.feature_utils import FeatureExtractor
from soundkit.utils.calculate_feat_stats import load_feat_stats
from soundkit.utils.tf_copy_model import copy_model_weights
from soundkit.utils.basic_dsp import DCRemover
from soundkit.utils.audio import audio_read
from soundkit.utils.calculate_feat_stats import mean_varinace_norm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
log = logging.getLogger(__name__)


def evaluate(params: SKTaskParams):
    """Evaluate ID task model with given parameters.

    Args:
        params (SKTaskParams): Task parameters

    """
    log.debug(f"Evaluating ID model with params: {params} and more")

    params_evaluate = params.evaluate

    checkpoint_dir = f"{params.train['path']['checkpoint_dir']}"
    batchsize=1

    dir = params.evaluate['data']['dir']

    # generate tfrecords for test files
    if params.data.signal.dc_removal:
        log.info("DC removal is enabled. Initializing DCRemover...")
        # Initialize DCRemover if DC removal is enabled
        dc_remover = DCRemover()
    else:
        dc_remover = None

    wavs_reg = [os.path.join(dir, f) for f in params.evaluate['data']['reg_files']]

    wavs_test = [os.path.join(dir, f) for f in params.evaluate['data']['test_files']]

    params.train['batchsize'] = params.data['ppls_per_group'] * params.data['num_sentences']
    batchsize_train = params.train['batchsize']


    # 1. Define feature extractor
    feat_extractor = FeatureExtractor(
        params=params,
    )
    dim_feat = feat_extractor.dim_feat

    # 2. Build model architecture
    # Load Model architecture from YAML file
    time_steps = int(params.data['target_length_in_secs'] * params.data.signal.sampling_rate) //  params.train.feature.hop_size
    model_train = build_model(
        params,
        batchsize=batchsize_train,
        dim_feat=dim_feat,
        time_steps=time_steps)

    # load weights from the checkpoint

    load_model_checkpoint(
        model_train, params_evaluate['epoch_loaded'], checkpoint_dir)

    # 3. Compute feature statistics for standardization

    stats = load_feat_stats(
        dir=checkpoint_dir,
        stats_name='stats.pkl')

    if params.evaluate.threshold_id > 1.0 or params.evaluate.threshold_id < 0.0:
        raise ValueError(
            f"Threshold for ID verification must be in the range [0.0, 1.0]. Provided: {params.evaluate.threshold_id}"
        )

    def cos_sim(a, b):
        """Compute cosine similarity between two vectors."""
        a_norm = tf.norm(a, axis=-1)
        b_norm = tf.norm(b, axis=-1)
        corr = tf.reduce_sum(a * b) / (a_norm * b_norm + 1e-8)
        return corr

    def spk_nn_inference(audio_sn):
        audio_sn = tf.convert_to_tensor(audio_sn, dtype=tf.float32)
        audio_sn = tf.expand_dims(audio_sn, axis=0)  # Add batch dimension

        feat_sn, *_ = feat_extractor(
            audio_sn)

        if params.train['standardization']:
            # Standardize features
            feat_sn_norm = mean_varinace_norm(feat_sn, stats['nMean_feat'], stats['nInvStd'])
        else:
            feat_sn_norm = feat_sn

        time_steps = feat_sn_norm.shape[1]

        model = build_model(
            params,
            batchsize=batchsize,
            dim_feat=dim_feat,
            time_steps=time_steps,
            summary=False)

        copy_model_weights(
            model_dst=model,
            model_src=model_train)

        out = model(feat_sn_norm, training=False)

        final_step = tf.math.minimum(179, time_steps - 1)
        d_vec = out[0, final_step, :]
        return d_vec


    # 4. Evaluate registration files
    log.info("Calculate the characteristics of registration files")
    d_vecs = []
    for step, wav_reg in enumerate(wavs_reg):
        print(f"\rEvaluating {wav_reg}, ", end='')
        # Read audio file
        audio_sn = audio_read(
            wav_reg,
            params.data['signal']['sampling_rate']
        )
        if dc_remover:
            # Apply DC removal if enabled
            audio_sn = dc_remover.process(audio_sn)
        d_vec = spk_nn_inference(audio_sn)
        d_vecs.append(d_vec)

    d_vecs = tf.stack(d_vecs)
    norm = tf.reduce_sum(d_vecs**2, axis=-1, keepdims=True)**0.5
    spk_reg = tf.reduce_mean(d_vecs / norm, axis=0)

    # 5. Now evaluate the test files
    for step, wav_test in enumerate(wavs_test):
        print(f"\rEvaluating {wav_test}, ", end='')
        # Read audio file
        audio_sn = audio_read(
            wav_test,
            params.data['signal']['sampling_rate']
        )

        if dc_remover:
            # Apply DC removal if enabled
            audio_sn = dc_remover.process(audio_sn)

        d_vec = spk_nn_inference(audio_sn)

        score = cos_sim(spk_reg, d_vec)

        if score < params_evaluate['threshold_id']:
            print(f"Test file {wav_test} is NOT recognized as registered speaker. (score={score.numpy():.4f} < threshold={params_evaluate['threshold_id']})")
        else:
            print(f"Test file {wav_test} is recognized as registered speaker. (score={score.numpy():.4f} >= threshold={params_evaluate['threshold_id']})")