''' datasets module is used to 
    
    1 create tfrecord files
    2 create tfrecord pipeline

'''
import logging
import os
from pathlib import Path
from typing import List, Tuple, Iterator
import tensorflow as tf
import numpy as np
from typing import List, Callable, Optional
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

        start = tf.random.uniform([],
                                minval=0,
                                maxval=len_raw - truncate_samples + 1,
                                dtype=tf.int32)

        audio_sn = audio_sn[start:start+truncate_samples]
        audio_s = audio_s[start:start+truncate_samples]
    return audio_sn, audio_s, length

def create_tfrecords_pipeline(
            filenames: List[str],
            batchsize: int = 2,
            num_per_epoch_files: Optional[int] = None,    # e.g., 40000 (random subset per epoch); None = use all files
            truncate_samples: Optional[int] = None,
            is_shuffle: bool = False) -> Tuple[Iterator, tf.data.Dataset]:
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
        is_shuffle: bool = False) -> Tuple[tf.data.Dataset, int]:
    """
    Create dataset from tfrecord list
    """
    
    if isinstance(tfrecords, (str, Path)):
        with open(tfrecords, 'r') as file: # pylint: disable=unspecified-encoding
            try:
                lines = file.readlines()

            except:# pylint: disable=bare-except
                log.warning('Can not find the list %s', tfrecords)
            else:
                create_tfrecords_pipeline
                total_batches = len(lines) // batchsize
                len0 = total_batches * batchsize
                fnames = [line.strip() for line in lines[:len0]]
                
                if num_per_epoch_files is not None:
                    if len(fnames) < num_per_epoch_files:
                        raise ValueError(
                            "num_per_epoch_files is larger than the total number of tfrecords.",
                            " Reduce num_per_epoch_files or add more tfrecords")
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
            is_shuffle = is_shuffle)

    if num_per_epoch_files is not None:
        total_batches = num_per_epoch_files // batchsize
    return dataset, total_batches
