import re
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import soundfile as sf
import matplotlib.pyplot as plt
from .datasets import create_raw_tfrecord
from .datasets import create_dataset
from ...defines import SKTaskParams
from ...utils.download_tf_model import build_model, load_model_checkpoint
from ...utils.feature_utils import FeatureExtractor
from ...utils.calculate_feat_stats import feat_stats_estimator
from ...utils.lookaheadBuffer import LookaheadBuffer

from ...utils.tf_copy_model import copy_model_weights
from ...utils.basic_dsp import dc_remove
from ...utils.audio import audio_read
from ...utils.plot_api import plot_spectrograms
from ...utils.tf_basic_math import tf_log10_eps
from ...utils.calculate_feat_stats import mean_varinace_norm

def evaluate(params: SKTaskParams):
    """Evaluate SE task model with given parameters.

    Args:
        params (HKTaskParams): Task parameters
    """
    print(f"Evaluating SE model with params: {params} and more")

    params_evaluate = params.evaluate
    num_lookahead = params.train['num_lookahead']
    feat_params = params.train['feature']
    checkpoint_dir = f"{params.train['path']['checkpoint_dir']}"
    result_folder = params.evaluate['data']['result_folder']
    batchsize=1

    dir = params.evaluate['data']['dir']
    dst_dir = f'{result_folder}/tfrecords'
    os.makedirs(dst_dir, exist_ok=True)

    # generate tfrecords for test files

    if params.evaluate['data']['files'] is None:
        wavs_path=[]
        wavs = []
        for root, dirs, files in os.walk(dir):
            for f in files:
                if re.search(r'(\.wav$|\.flac$)', f):
                    wavs += [f]
                    wavs_path += [os.path.join(root, f)]
    else:
        wavs = params.evaluate['data']['files']
        wavs_path = [os.path.join(dir, f) for f in params.evaluate['data']['files']]

    tfrecords=[]

    for wav_path in tqdm(wavs_path, desc="Generating TFRecords", unit="file"):
        fname = os.path.basename(wav_path)
        tfname= re.sub(r'(\.wav$|\.flac$)', '.tfrecord', fname)
        tfrecord= f"{dst_dir}/{tfname}"

        sig = audio_read(
            wav_path,
            params.data['signal']['sampling_rate']
        )

        if params.data['signal']['dc_removal']:
            sig = dc_remove(sig)

        create_raw_tfrecord(
            tfrecord, sig, (np.array([]),np.array([])))
        tfrecords += [tfrecord]
        dataset, batches = create_dataset(
            tfrecords,
            batchsize=batchsize,
            is_shuffle=False,
        )

    batchsize_train = params.train['batchsize']


    # 1. Define feature extractor
    feat_extractor = FeatureExtractor(
        params=params,
    )
    dim_feat = feat_extractor.dim_feat

    # 2. Build model architecture
    # Load Model architecture from YAML file

    model_train = build_model(
        params,
        batchsize=batchsize_train,
        dim_feat=dim_feat,
        time_steps = params.data['target_length_in_secs'] * 100)

    # load weights from the checkpoint

    load_model_checkpoint(
        model_train, params_evaluate['epoch_loaded'], checkpoint_dir)

    # 3. Compute feature statistics for standardization
    stats = feat_stats_estimator(
        dataset,
        batches,
        folder_nn=checkpoint_dir,
        feat_extractor=feat_extractor,)

    for step, batch in enumerate(dataset):
        print(f"\rEvaluating (batch) {step}/{batches}, ", end='')

        # Initialize left-over state buffers for streaming STFT
        states_audio_sn = tf.zeros(
            [batchsize, feat_params["frame_size"] - feat_params["hop_size"]],
            dtype=tf.float32
        )

        audio_sn, *_ = batch

        feat_sn, spec_sn, states_audio_sn = feat_extractor(
            audio_sn, states=states_audio_sn)

        if params.train['standardization']:
            # Standardize features
            feat_sn_norm = mean_varinace_norm(feat_sn, stats['nMean_feat'], stats['nInvStd'])
        else:
            feat_sn_norm = feat_sn

        time_steps = feat_sn_norm.shape[1]
        if 0:
            tfmask = []
            for f in range(time_steps):
                print(f"\rframe {f}/{time_steps}, ", end='')
                feat = feat_sn_norm[:, f:f+1, :]
                m = model(feat, training=False)
                tfmask+= [m]
            tfmask = tf.concat(tfmask, axis=1)

        else:
            model = build_model(
                params,
                batchsize=batchsize,
                dim_feat=dim_feat,
                time_steps=time_steps)

            copy_model_weights(
                model_dst=model,
                model_src=model_train)

            out = model(feat_sn_norm, training=False)
            out = tf.math.softmax(out, axis=-1)
            prob = out[...,1]
            vad = tf.cast(prob > 0.5, dtype=tf.int32)

            stride = model.stride_time

            hop_size = feat_params['hop_size']

        if step < 10:
            # draw spectrograms and tfmask
            name = re.sub(r'(\.wav$|\.flac$)', '.pdf', wavs[step])
            save_path = f"{result_folder}/{name}"

            if params.train['feature']['type'] in ('mel', 'logpspec', 'hybrid'):
                logpfeat_sn = (10 * feat_sn).numpy()[0].T
            elif params.train['feature']['type'] in ('pspec'):
                logpfeat_sn = 10*tf_log10_eps( tf.abs(feat_sn[0])).numpy()
            elif params.train['feature']['type'] in ('spec'):
                logpfeat_sn = 20*tf_log10_eps( tf.abs(feat_sn[0])).numpy()
            logpspec_sn = 20 * tf_log10_eps(tf.abs(spec_sn)).numpy()[0].T

            plot_spectrograms(
                images=[
                    logpfeat_sn,
                    logpspec_sn,
                ],
                titles=["Noisy feat", "Noisy pspec"],
                vmin_vmax=[(-80, 10), (-80, 10)],
                save_path=None,
            )

            vad = vad[0]
            vad = tf.tile(tf.expand_dims(vad, axis=0), [stride, 1])    
            vad = tf.reshape(tf.transpose(vad), [-1])

            prob = prob[0]
            prob = tf.tile(tf.expand_dims(prob, axis=0), [stride, 1])
            prob = tf.reshape(tf.transpose(prob), [-1])
            
            plt.plot(vad*250)
            plt.plot(prob*250)
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
            print(f"Saved figure to {save_path}")




            # Save noisy audio
            sample_level_vad = tf.repeat(vad.numpy(), repeats=hop_size)
            sample_level_vad = tf.cast(sample_level_vad, dtype=tf.float32)

            name = re.sub(r'(\.wav$|\.flac$)', '_sn.wav', wavs[step])
            save_path = f"{result_folder}/{name}"

            audio_sn_np = tf.squeeze(audio_sn, axis=0).numpy()
            sf.write(
                save_path,
                audio_sn_np,
                params.data['signal']['sampling_rate'])
            print(f"Saved noisy audio to {save_path}")

            # save enhanced audio
            name = re.sub(r'(\.wav$|\.flac$)', '_vad.wav', wavs[step])
            save_path = f"{result_folder}/{name}"
            audio_vad_np = sample_level_vad.numpy()
            sf.write(
                save_path,
                audio_vad_np,
                params.data['signal']['sampling_rate'])
            
            print(f"Saved vad audio to {save_path}")
        