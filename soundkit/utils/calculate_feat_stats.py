"""
Calculating statistic mean and standard deviation
"""
import os
import pickle
import tensorflow as tf
import numpy as np
from .converter_fix_point import fakefix_tf
from .feature_utils import FeatureExtractor

def load_feat_stats(dir: str,stats_name: str = 'stats.pkl'):

    if os.path.exists(os.path.join(dir, stats_name)):
        with open(os.path.join(dir, stats_name), "rb") as file:
            stats = pickle.load(file)
        return stats

def feat_stats_estimator(
        dataset: tf.data.Dataset,
        num_batches: int,
        folder_nn: str,
        feat_extractor: FeatureExtractor,)->None:
    """
    Estimate statistics of training data
    """
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
    for i, batch in enumerate(dataset):
        if i % 5 == 0:
            tf.print(f"\rMean estimating (batch) {i}/{num_batches}, ",
                        end = '')
        audio_sn = batch[0]

        feat_sn, _, _ = feat_extractor(
            audio_sn)

        if tf.as_dtype(feat_sn.dtype).is_complex:
            feat_sn = tf.math.abs(feat_sn)
        shape = tf.shape(feat_sn)
        tmp = tf.math.reduce_sum(feat_sn, axis = (0,1))
        mean_stats = mean_stats + tf.cast(tmp, tf.float64)
        tmp = shape[0] * shape[1]
        tot = tot + tf.cast(tmp, tf.float64)

    mean_stats = mean_stats / tot
    mean_stats = tf.cast(mean_stats, tf.float32)
    mean_stats = fakefix_tf(mean_stats, 32, 15)

    # std calculation
    for i, batch in enumerate(dataset):
        if i % 5 == 0:
            tf.print(f"\rSTD estimating (batch) {i}/{num_batches}, ",
                        end = '')
        audio_sn = batch[0]
        feat_sn, _,_ = feat_extractor(
            audio_sn)
        if tf.as_dtype(feat_sn.dtype).is_complex:
            feat_sn = tf.math.abs(feat_sn)
        shape = tf.shape(feat_sn)
        tmp = tf.math.reduce_sum((feat_sn - mean_stats)**2, axis = (0,1))
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
