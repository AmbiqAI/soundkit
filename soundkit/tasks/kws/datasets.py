''' datasets module is used to 
    
    1 create tfrecord files
    2 create tfrecord pipeline

'''
import os
import logging
from pathlib import Path
from typing import List, Tuple, Iterator
import tensorflow as tf
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
log = logging.getLogger(__name__)

def create_raw_tfrecord(
        fname: str,
        audio_sn: np.ndarray,
        indices: tuple[np.ndarray, np.ndarray]) -> None:
    """
    Make TFRecord with multiple start/end indices
    """
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with tf.io.TFRecordWriter(fname) as writer:
        timesteps = audio_sn.shape[0]
        start_indices, end_indices = indices  # both are np.ndarray or list

        audio_sn_feature = tf.train.Feature(
            float_list=tf.train.FloatList(value=audio_sn))

        context = tf.train.Features(feature={
            "length": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[timesteps])),
            "start_index": tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(start_indices))),
            "end_index": tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(end_indices))),
        })

        feature_lists = tf.train.FeatureLists(feature_list={
            "audio_sn": tf.train.FeatureList(
                feature=[audio_sn_feature]
            )
        })

        seq_example = tf.train.SequenceExample(
            context=context,
            feature_lists=feature_lists
        )

        writer.write(seq_example.SerializeToString())

def parser(example_proto: tf.Tensor, hop_size: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Parse a TFRecord sequence example with multi-segment KWS labels.

    Returns:
        audio_sn: 1D waveform
        length: total sample length
        KWS: frame-wise KWS labels (0 or 1)
    """
    # Context: global metadata
    context_features = {
        'length': tf.io.FixedLenFeature([], tf.int64),
        'start_index': tf.io.VarLenFeature(tf.int64),
        'end_index': tf.io.VarLenFeature(tf.int64),
    }

    # Sequence: per-frame feature
    sequence_features = {
        'audio_sn': tf.io.VarLenFeature(tf.float32),
    }

    # Parse sequence example
    context_parsed, seq_parsed = tf.io.parse_single_sequence_example(
        example_proto,
        context_features=context_features,
        sequence_features=sequence_features,
    )

    # Waveform: convert from sparse
    audio_sn = tf.sparse.to_dense(seq_parsed['audio_sn'])

    # Total number of frames
    total_length = context_parsed['length']
    num_frames = tf.cast(total_length // hop_size, tf.int32)

    # Convert sparse KWS start/end to dense
    start_indices = tf.sparse.to_dense(context_parsed['start_index'])
    end_indices = tf.sparse.to_dense(context_parsed['end_index'])

    start_frames = tf.cast(start_indices // hop_size, tf.int32)
    end_frames = tf.cast(end_indices // hop_size, tf.int32)

    # Clip frame boundaries
    start_frames = tf.maximum(start_frames, 0)
    end_frames = tf.minimum(end_frames, num_frames)

    # Frame-level KWS label (vectorized scatter)
    kws = tf.zeros((num_frames,), dtype=tf.int32)

    # Create ragged index ranges
    ragged_ranges = tf.ragged.range(start_frames, end_frames)
    flat_indices = ragged_ranges.flat_values
    updates = tf.ones_like(flat_indices, dtype=tf.int32)

    indices = tf.expand_dims(flat_indices, axis=1)
    kws = tf.tensor_scatter_nd_update(kws, indices, updates)

    return (
        audio_sn[0],  # return 1D waveform
        tf.cast(total_length, tf.int32),
        kws,
    )
def create_tfrecords_pipeline(
            filenames: List[str],
            hop_size: int = 160,
            batchsize: int = 2,
            is_shuffle: bool = False) -> Tuple[Iterator, tf.data.Dataset]:
    """
    Tfrecord generator
    """
    def mapping(record):
        return parser(record, hop_size)

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
        hop_size: int = 160,
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
            hop_size=hop_size,
            is_shuffle = is_shuffle)

    return dataset, total_batches
