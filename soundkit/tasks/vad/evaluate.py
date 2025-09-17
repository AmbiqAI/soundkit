import re
import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import soundfile as sf
import matplotlib.pyplot as plt
from soundkit.defines import SKTaskParams
from soundkit.utils.download_tf_model import build_model, load_model_checkpoint
from soundkit.utils.feature_utils import FeatureExtractor
from soundkit.utils.calculate_feat_stats import load_feat_stats
from soundkit.utils.tf_copy_model import copy_model_weights
from soundkit.utils.basic_dsp import dc_remove
from soundkit.utils.audio import audio_read
from soundkit.utils.plot_api import plot_spectrograms
from soundkit.utils.tf_basic_math import tf_log10_eps
from soundkit.utils.calculate_feat_stats import mean_varinace_norm

def evaluate(params: SKTaskParams):
    """Evaluate VAD task model with given parameters.

    Args:
        params (HKTaskParams): Task parameters

    """
    from .utils.vad_silero import get_vad, calculate_vad_accuracy
    print(f"Evaluating VAD model with params: {params} and more")

    params_evaluate = params.evaluate
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
        time_steps = params.data['target_length_in_secs'] * params.data.signal.sampling_rate //  params.train.feature.hop_size)

    # load weights from the checkpoint

    load_model_checkpoint(
        model_train, params_evaluate['epoch_loaded'], checkpoint_dir)

    # 3. Compute feature statistics for standardization
    
    stats = load_feat_stats(
        dir=checkpoint_dir,
        stats_name='stats.pkl')


    for step, wavs_path in enumerate(wavs_path):
        print(f"\rEvaluating {wavs_path}, ", end='')
        # Read audio file
        audio_sn_np = audio_read(
            wavs_path,
            params.data['signal']['sampling_rate']
        )

        audio_sn = tf.convert_to_tensor(audio_sn_np, dtype=tf.float32) 
        audio_sn = tf.expand_dims(audio_sn, axis=0)  # Add batch dimension


        feat_sn, spec_sn, states_audio_sn = feat_extractor(
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
            time_steps=time_steps)

        copy_model_weights(
            model_dst=model,
            model_src=model_train)

        out = model(feat_sn_norm, training=False)
        out = tf.math.softmax(out, axis=-1)
        prob = out[...,1]
        vad = tf.cast(prob > 0.5, dtype=tf.int32)

        if hasattr(model, 'stride_time'):
            stride = model.stride_time
        else:
            stride = 1
        hop_size = feat_params['hop_size']


        # draw spectrograms and tfmask
        name = re.sub(r'(\.wav$|\.flac$)', '.pdf', wavs[step])
        save_path = f"{result_folder}/{name}"

        if params.train['feature']['type'] in ('mel', 'logpspec', 'hybrid'):
            logpfeat_sn = (10 * feat_sn).numpy()[0].T
        elif params.train['feature']['type'] in ('pspec'):
            logpfeat_sn = 10*tf_log10_eps( tf.abs(feat_sn[0])).numpy().T
        elif params.train['feature']['type'] in ('spec'):
            logpfeat_sn = 20*tf_log10_eps( tf.abs(feat_sn[0])).numpy().T
        elif params.train['feature']['type'] in ('time'):
            logpfeat_sn = 20*tf_log10_eps( tf.abs(spec_sn[0])).numpy().T
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
        plt.legend(['VAD outputs', 'Prob of speech'])
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
        print(f"Saved original audio to {save_path}")

        # save enhanced audio
        name = re.sub(r'(\.wav$|\.flac$)', '_vad.wav', wavs[step])
        save_path = f"{result_folder}/{name}"
        audio_vad_np = sample_level_vad.numpy()
        sf.write(
            save_path,
            audio_vad_np,
            params.data['signal']['sampling_rate'])

        print(f"Saved vad signal to {save_path}")
        
        # Silero VAD
        print("Calculating silero vad")
        vad_gt = get_vad(audio_sn_np, sampling_rate=16000) # silero vad
        name = re.sub(r'(\.wav$|\.flac$)', '_vad_silero.wav', wavs[step])
        save_path = f"{result_folder}/{name}"
        sf.write(
            save_path,
            vad_gt,
            params.data['signal']['sampling_rate'])

        print(f"Saved silero vad (ground truth) VAD to {save_path}")


        vad = audio_vad_np
        vad_gt = vad_gt[:len(vad)]  # Ensure both arrays are the same length

        vad = (vad > 0.1).astype(np.int16)
        vad_gt = vad_gt.astype(np.int16)
        fa, fr, fa_total, fr_total = calculate_vad_accuracy(vad, vad_gt)

        print(f"False Alarms: {fa:.4f} (total: {fa_total})")
        print(f"False Rejections: {fr:.4f} (total: {fr_total})")
