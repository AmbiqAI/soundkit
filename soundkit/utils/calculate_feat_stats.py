"""
Calculating statistic mean and standard deviation
"""
import os
import pickle
from tqdm import tqdm
import tensorflow as tf
from soundkit.utils.converter_fix_point import fakefix_tf
from soundkit.utils.feature_utils import FeatureExtractor

def load_feat_stats(dir: str,stats_name: str = 'stats.pkl'):

    if os.path.exists(os.path.join(dir, stats_name)):
        with open(os.path.join(dir, stats_name), "rb") as file:
            stats = pickle.load(file)
        return stats

@tf.function
def calculate_mean(feat_sn, mask = 1):
    if tf.as_dtype(feat_sn.dtype).is_complex:
        feat_sn = tf.math.abs(feat_sn * mask)

    sub_tot = tf.math.reduce_sum(feat_sn * mask, axis = (0,1))
    return sub_tot


@tf.function
def calculate_std(feat_sn, mean_stats, mask=1):
    """
    Calculate the standard deviation
    """
    if tf.as_dtype(feat_sn.dtype).is_complex:
        feat_sn = tf.math.abs(feat_sn)
    sub_tot = tf.math.reduce_sum(mask * (feat_sn - mean_stats)**2, axis = (0,1))
    return sub_tot

def feat_stats_estimator(
        dataset: tf.data.Dataset,
        num_batches: int,
        folder_nn: str,
        feat_extractor: FeatureExtractor=None,
        dim_feat=40)->None:
    """
    Estimate statistics of training data
    """

    if feat_extractor is None:
        pass
    else:
        dim_feat = feat_extractor.dim_feat
    stats_name = 'stats.pkl'
    if os.path.exists(os.path.join(folder_nn, stats_name)):
        return load_feat_stats(folder_nn, stats_name)

    mean_stats = tf.Variable( tf.zeros((dim_feat,), dtype = tf.float64),
                    dtype = tf.float64, trainable = False)
    inv_std_stats = tf.Variable(tf.zeros((dim_feat,), dtype = tf.float64),
                        dtype = tf.float64, trainable = False)
    tot = tf.Variable(0, dtype = tf.float64, trainable=False)

    # mean calculation
    for i, batch in enumerate(tqdm(dataset, total=num_batches, desc="Mean estimating (batch)")):
        if feat_extractor is None:
            feat_sn = batch[0]
            mask = batch[-1]
        else:
            audio_sn = batch[0]
            mask = 1
            feat_sn, *_ = feat_extractor(
                audio_sn)

        sub_tot = calculate_mean(feat_sn, mask)
        mean_stats = mean_stats + tf.cast(sub_tot, tf.float64)
        shape = tf.shape(feat_sn)
        tmp = shape[0] * shape[1]
        tot = tot + tf.cast(tmp, tf.float64)

    mean_stats = mean_stats / tot
    mean_stats = tf.cast(mean_stats, tf.float32)
    mean_stats = fakefix_tf(mean_stats, 32, 15)

    # std calculation
    for i, batch in enumerate(tqdm(dataset, total=num_batches, desc="STD estimating (batch)")):
        if feat_extractor is None:

            feat_sn = batch[0]
            mask = batch[-1]
        else:
            audio_sn = batch[0]
            mask = 1
            feat_sn, *_ = feat_extractor(
                audio_sn)
        sub_tot = calculate_std(feat_sn, mean_stats, mask)

        inv_std_stats = inv_std_stats + tf.cast(tmp, tf.float64)

    inv_std_stats = 1.0 / (2**-15 + tf.math.sqrt(inv_std_stats / tot))
    inv_std_stats = tf.cast(inv_std_stats, tf.float32)
    inv_std_stats = fakefix_tf(inv_std_stats, 32, 15)

    # save mean and std
    stats = {'nMean_feat': mean_stats.numpy(), 'nInvStd': inv_std_stats.numpy()}

    os.makedirs(folder_nn, exist_ok=True)
    with open(os.path.join(folder_nn, stats_name), "wb") as file:
        pickle.dump(stats, file)

    return stats

def mean_varinace_norm(
        inputs: tf.Tensor,
        mean_stats: tf.Tensor | None = None,
        inv_std_stats: tf.Tensor | None = None) -> tf.Tensor:
    """
    Normalize the input features using the provided mean and inverse standard deviation.
    """

    if tf.as_dtype(inputs.dtype).is_complex:
        mag = tf.abs(inputs)
        phase = tf.math.angle(inputs)
        norm_mag = (mag - mean_stats) * inv_std_stats
        outputs = tf.cast(norm_mag, tf.complex64) * tf.exp(1j * tf.cast(phase, tf.complex64))
    else:
        outputs = (inputs - mean_stats) * inv_std_stats

    return outputs
