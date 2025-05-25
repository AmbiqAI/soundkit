''' datasets module is used to 
    
    1 create tfrecord files
    2 create tfrecord pipeline

'''
import os
from pathlib import Path
from typing import List, Tuple, Iterator
import tensorflow as tf
import numpy as np

def create_raw_tfrecord(
        fname: str,
        audio_sn: np.ndarray,
        audio_s: np.ndarray) -> None:
    """
    Make tfrecord
    """
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with tf.io.TFRecordWriter(fname) as writer:

        timesteps = audio_s.shape[0]

        step_feature = tf.train.Feature(
            int64_list = tf.train.Int64List(value = [timesteps]))

        audio_sn_feature = tf.train.Feature(
            float_list = tf.train.FloatList(value = audio_sn))

        audio_s_feature = tf.train.Feature(
            float_list = tf.train.FloatList(value = audio_s))

        context = tf.train.Features(feature = {
                "length"    : step_feature,
            })

        feature_lists = tf.train.FeatureLists(feature_list={
                "audio_sn" : tf.train.FeatureList(feature = [audio_sn_feature]),
                "audio_s"  : tf.train.FeatureList(feature = [audio_s_feature]),
            })

        seq_example = tf.train.SequenceExample( # context and feature_lists
            context = context,
            feature_lists = feature_lists,
        )

        serialized = seq_example.SerializeToString()
        writer.write(serialized)

def parser( example_proto: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Create a description of the features.
    """
    context_features = {
        'length'    : tf.io.FixedLenFeature([], tf.int64),
    }

    sequence_features = {
        'audio_sn'      : tf.io.VarLenFeature(tf.float32),
        'audio_s'       : tf.io.VarLenFeature(tf.float32),
    }
    context_parsed, seq_parsed = tf.io.parse_single_sequence_example(
            example_proto,
            context_features  = context_features,
            sequence_features = sequence_features,
                                        )

    length = tf.cast(context_parsed['length'], tf.int32)

    audio_sn = tf.sparse.to_dense(seq_parsed['audio_sn'])

    audio_s = tf.sparse.to_dense(seq_parsed['audio_s'])

    return audio_sn[0], audio_s[0], length

def create_tfrecords_pipeline(
            filenames: List[str],
            batchsize: int = 2,
            is_shuffle: bool = False) -> Tuple[Iterator, tf.data.Dataset]:
    """
    Tfrecord generator
    """
    def mapping(record):
        return parser(record)

    def tfrecord_convert(val):
        return tf.data.TFRecordDataset(val)

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if is_shuffle:
        dataset = dataset.shuffle(len(filenames), reshuffle_each_iteration=True)
    dataset = dataset.interleave(
                map_func           = tfrecord_convert,
                cycle_length       = batchsize,
                block_length       = 1,
                deterministic      = True,
                num_parallel_calls = tf.data.AUTOTUNE)
    dataset = dataset.map(
                mapping,
                num_parallel_calls = tf.data.AUTOTUNE,
                deterministic = True)
    dataset = dataset.batch(
                    batchsize,
                    drop_remainder=True,
                    num_parallel_calls = tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size = 1)
    iterator = iter(dataset)
    return iterator, dataset

def create_dataset(
        tfrecords: str | list,
        batchsize: int = 2,
        is_shuffle: bool = False) -> Tuple[tf.data.Dataset, int]:
    """
    Create dataset from tfrecord list
    """
    
    if isinstance(tfrecords, (str, Path)):
        with open(tfrecords, 'r') as file: # pylint: disable=unspecified-encoding
            try:
                lines = file.readlines()

            except:# pylint: disable=bare-except
                print(f'Can not find the list {tfrecords}')
            else:
                
                total_batches = len(lines) // batchsize
                len0 = total_batches * batchsize
                fnames = [line.strip() for line in lines[:len0]]
                # if num_samples > 0:
                #     import random
                #     random.seed(42)
                #     random.shuffle(fnames[tr_set])
                #     if tr_set=='train':
                #         fnames[tr_set] = fnames[tr_set][:num_samples]
                #     else:
                #         fnames[tr_set] = fnames[tr_set][:num_samples >> 2]
    elif isinstance(tfrecords, list):
        fnames = tfrecords
        total_batches = len(fnames) // batchsize
    else:
        raise ValueError("tfrecords should be a string or a list of strings.")

    _, dataset = create_tfrecords_pipeline(
            fnames,
            batchsize = batchsize,
            is_shuffle = is_shuffle)

    return dataset, total_batches
