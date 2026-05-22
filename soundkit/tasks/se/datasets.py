''' datasets module is used to 
    
    1 create tfrecord files
    2 create tfrecord pipeline

'''
import logging
import re
import os
from pathlib import Path
from typing import List, Tuple, Iterator, Optional
import numpy as np
import tensorflow as tf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
log = logging.getLogger(__name__)
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

def parser(
        example_proto: tf.Tensor,
        truncate_samples: Optional[int] = None,
        ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
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

    audio_sn = audio_sn[0]
    audio_s = audio_s[0]
    if truncate_samples is not None:
        len_raw = tf.shape(audio_sn)[0]
        keep_samples = tf.where(
            len_raw >= truncate_samples,
            truncate_samples,
            tf.maximum(tf.cast(tf.floor(tf.cast(len_raw, tf.float32) * 0.95), tf.int32), 1))

        start = tf.random.uniform([],
                                minval=0,
                                maxval=len_raw - keep_samples + 1,
                                dtype=tf.int32)

        audio_sn = audio_sn[start:start+keep_samples]
        audio_s = audio_s[start:start+keep_samples]
        pad_len = truncate_samples - keep_samples
        audio_sn = tf.pad(audio_sn, [[0, pad_len]])
        audio_s = tf.pad(audio_s, [[0, pad_len]])
        length = keep_samples
    return audio_sn, audio_s, length

def create_tfrecords_pipeline(
            filenames: List[str],
            batchsize: int = 2,
            num_per_epoch_files: Optional[int] = None,    # e.g., 40000 (random subset per epoch); None = use all files
            truncate_samples: Optional[int] = None,
            is_shuffle: bool = False,
            seed: Optional[int] = None) -> Tuple[Iterator, tf.data.Dataset]:
    """
    Tfrecord generator
    """
    def mapping(record):
        return parser(record, truncate_samples=truncate_samples)

    def tfrecord_convert(val):
        return tf.data.TFRecordDataset(val)

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if is_shuffle:
        dataset = dataset.shuffle(
            len(filenames),
            seed=seed,
            reshuffle_each_iteration=True)
    if num_per_epoch_files is not None:
        dataset = dataset.take(num_per_epoch_files)
    dataset = dataset.interleave(
                map_func           = tfrecord_convert,
                cycle_length       = batchsize,
                block_length       = 1,
                deterministic      = True,
                num_parallel_calls = tf.data.AUTOTUNE,)
    dataset = dataset.map(
                mapping,
                num_parallel_calls = 1,
                deterministic = True)
    dataset = dataset.batch(
                    batchsize,
                    drop_remainder=True,
                    num_parallel_calls = 1)
    dataset = dataset.prefetch(buffer_size = 1)
    iterator = iter(dataset)
    return iterator, dataset

def create_dataset(
        tfrecords: str | list,
        batchsize: int = 2,
        num_per_epoch_files: Optional[int] = None,    # e.g., 40000 (random subset per epoch); None = use all files
        truncate_samples: Optional[int] = None,
        is_shuffle: bool = False,
        seed: Optional[int] = None) -> Tuple[tf.data.Dataset, int]:
    """
    Create dataset from tfrecord list
    """

    if isinstance(tfrecords, (str, Path)):
        with open(tfrecords, 'r') as file: # pylint: disable=unspecified-encoding
            try:
                lines = file.readlines()
                if 1: # only use mandarin files
                    import random as _rnd
                    import random
                    if is_shuffle:
                        en = []
                        ch = []
                        for line in lines:
                            if re.search(r'(LibriSpeech|german_speech|spanish_speech|italian_speech|french_data)', line):
                                en.append(line)
                            elif re.search(r'(mandarin|MAGICDATA)', line):
                                ch.append(line)
                        # import pdb; pdb.set_trace()
                        random.seed(0)
                        random.shuffle(en)
                        random.shuffle(ch)
                        lines = ch+en

            except:# pylint: disable=bare-except
                log.warning('Can not find the list %s', tfrecords)
            else:

                total_batches = len(lines) // batchsize
                len0 = total_batches * batchsize
                fnames = [line.strip() for line in lines[:len0]]

                if num_per_epoch_files is not None:
                    if len(fnames) < num_per_epoch_files:
                        print(f"Warning: num_per_epoch_files ({num_per_epoch_files}) is larger than the number of available files ({len(fnames)}). Using all available files.")
                        fnames = fnames[:num_per_epoch_files]
                        num_per_epoch_files = len(fnames)
                        total_batches = num_per_epoch_files // batchsize
                        len0 = total_batches * batchsize
                        fnames = [line.strip() for line in lines[:len0]]

    elif isinstance(tfrecords, list):
        fnames = tfrecords
        total_batches = len(fnames) // batchsize
    else:
        raise ValueError("tfrecords should be a string or a list of strings.")

    _, dataset = create_tfrecords_pipeline(
            fnames,
            batchsize = batchsize,
            num_per_epoch_files = num_per_epoch_files,
            truncate_samples = truncate_samples,
            is_shuffle = is_shuffle,
            seed = seed)

    if num_per_epoch_files is not None:
        total_batches = num_per_epoch_files // batchsize
    return dataset, total_batches
