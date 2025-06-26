''' datasets module is used to 
    
    1 create tfrecord files
    2 create tfrecord pipeline

'''
import os
from pathlib import Path
from typing import List, Tuple, Iterator
import tensorflow as tf
import numpy as np
import yaml

def create_tfrecord(
        fname: str,
        feature: np.ndarray,
        indices: tuple[np.ndarray, np.ndarray]) -> None:
    """
    Make TFRecord with multiple start/end indices
    """
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with tf.io.TFRecordWriter(fname) as writer:
        num_frames, _ = feature.shape
        feature = feature.reshape([-1])
        start_indices, end_indices = indices  # both are np.ndarray or list

        feature = tf.train.Feature(float_list = tf.train.FloatList(value = feature))


        context = tf.train.Features(feature={
            "num_frames": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[num_frames])),
            "start_frame": tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(start_indices))),
            "end_frame": tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(end_indices))),
        })

        feature_lists = tf.train.FeatureLists(feature_list={
            "feat_sn": tf.train.FeatureList(
                feature=[feature]
            )
        })

        seq_example = tf.train.SequenceExample(
            context=context,
            feature_lists=feature_lists
        )

        writer.write(seq_example.SerializeToString())

# def create_raw_tfrecord(
#         fname: str,
#         audio_sn: np.ndarray,
#         indices: tuple[np.ndarray, np.ndarray]) -> None:
#     """
#     Make TFRecord with multiple start/end indices
#     """
#     os.makedirs(os.path.dirname(fname), exist_ok=True)
#     with tf.io.TFRecordWriter(fname) as writer:
#         timesteps = audio_sn.shape[0]
#         start_indices, end_indices = indices  # both are np.ndarray or list

#         audio_sn_feature = tf.train.Feature(
#             float_list=tf.train.FloatList(value=audio_sn))

#         context = tf.train.Features(feature={
#             "length": tf.train.Feature(
#                 int64_list=tf.train.Int64List(value=[timesteps])),
#             "start_index": tf.train.Feature(
#                 int64_list=tf.train.Int64List(value=list(start_indices))),
#             "end_index": tf.train.Feature(
#                 int64_list=tf.train.Int64List(value=list(end_indices))),
#         })

#         feature_lists = tf.train.FeatureLists(feature_list={
#             "audio_sn": tf.train.FeatureList(
#                 feature=[audio_sn_feature]
#             )
#         })

#         seq_example = tf.train.SequenceExample(
#             context=context,
#             feature_lists=feature_lists
#         )

#         writer.write(seq_example.SerializeToString())

def parser(example_proto: tf.Tensor, hop_size: int = 160) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Parse a TFRecord sequence example with multi-segment VAD labels.

    Returns:
        audio_sn: 1D waveform
        num_frames: total sample length
        vad: frame-wise VAD labels (0 or 1)
    """
    # Context: global metadata
    context_features = {
        'num_frames': tf.io.FixedLenFeature([], tf.int64),
        'start_frame': tf.io.VarLenFeature(tf.int64),
        'end_frame': tf.io.VarLenFeature(tf.int64),
    }

    # Sequence: per-frame feature
    sequence_features = {
        'feat_sn': tf.io.VarLenFeature(tf.float32),
    }

    # Parse sequence example
    context_parsed, seq_parsed = tf.io.parse_single_sequence_example(
        example_proto,
        context_features=context_features,
        sequence_features=sequence_features,
    )

    # Total number of frames
    num_frames = context_parsed['num_frames']
    num_frames = tf.cast(num_frames, tf.int32)
    # Waveform: convert from sparse
    feature = tf.sparse.to_dense(seq_parsed['feat_sn'])
    feature = tf.reshape(feature, (num_frames, -1))


    # Convert sparse VAD start/end to dense
    start_frame = tf.sparse.to_dense(context_parsed['start_frame'])
    end_frame = tf.sparse.to_dense(context_parsed['end_frame'])

    start_frames = tf.cast(start_frame, tf.int32)
    end_frames = tf.cast(end_frame, tf.int32)

    ones = tf.ones(end_frames[0] - start_frames[0] + 1)
    zeros = tf.zeros(num_frames - tf.shape(ones)[0])
    mask = tf.concat([ones, zeros], 0)
    mask = tf.cast(mask, tf.float32)
    mask = tf.expand_dims(mask,-1)


    return (
        feature,  # return 1D waveform
        tf.cast(num_frames, tf.int32),
        start_frames[0], end_frames[0],  # return start and end frames as tensors,
        mask
    )

def create_tfrecords_pipeline(
            filenames: List[str],
            num_sentences: int = 10,
            ppls_per_group: int = 64,
            num_utterances_in_sentence: int = 20,
            hop_size: int = 160) -> Tuple[Iterator, tf.data.Dataset]:
    """
    Tfrecord generator
    """

    num_ppls = len(filenames)


    def mapping_utterances(dataset):
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        dataset = dataset.shuffle(buffer_size=num_utterances_in_sentence)

        return dataset

    def mapping_ppl(dataset):
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        dataset = dataset.interleave( # interleave inside each speaker
            mapping_utterances,
            cycle_length=num_sentences,  # len(fnames)
            block_length=1,
            deterministic=True,
            num_parallel_calls=1
        )
        return dataset

    def mapping_shuffle(dataset):
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        dataset = dataset.shuffle(buffer_size=num_ppls)
        return dataset

    def decode_tfrecord(tfrecord_path):
        return tf.data.TFRecordDataset(tfrecord_path).map(
            lambda x: parser(x, hop_size=hop_size),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True
        )

    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    dataset = dataset.interleave( # interleave all speakers
            mapping_ppl,
            cycle_length=num_ppls, # len(fnames)
            block_length=num_sentences,
            deterministic=True,
            num_parallel_calls = 1)

    dataset = dataset.batch(num_sentences)

    dataset = dataset.batch(num_ppls)

    dataset = dataset.interleave(
            mapping_shuffle,
            cycle_length=1, # len(fnames)
            block_length=ppls_per_group,
            deterministic=True,
            num_parallel_calls = tf.data.AUTOTUNE)

    dataset = dataset.unbatch()

    dataset = dataset.interleave(
        decode_tfrecord,
        cycle_length=ppls_per_group * num_sentences,
        block_length=1,
        deterministic=True,
        num_parallel_calls=4
    )

    dataset = dataset.batch(ppls_per_group * num_sentences)

    dataset = dataset.prefetch(buffer_size = 1)

    iterator = iter(dataset)

    return iterator, dataset

def create_dataset(
        tfrecords: str | list,
        num_sentences: int = 10,
        ppls_per_group: int = 64,
        num_utterances_in_sentence: int = 20,
        hop_size: int = 160,
        ) -> Tuple[tf.data.Dataset, int]:
    """
    Create dataset from tfrecord list
    """
    if isinstance(tfrecords, (str, Path)):
        with open(tfrecords, "r") as file: # pylint: disable=unspecified-encoding
            try:
                fnames = yaml.safe_load(file)

            except:# pylint: disable=bare-except
                print(f'Can not find the list {tfrecords}')
            else:
                num_ppls = len(fnames)
                tot_sentences = (num_ppls * num_sentences * num_utterances_in_sentence)
                total_batches = tot_sentences // (ppls_per_group* num_sentences)
    else:
        raise ValueError("tfrecords should be a string or a list of strings.")

    _, dataset = create_tfrecords_pipeline(
            fnames,
            ppls_per_group=ppls_per_group,
            num_sentences=num_sentences,
            num_utterances_in_sentence=num_utterances_in_sentence,
            hop_size=hop_size,)

    return dataset, total_batches
