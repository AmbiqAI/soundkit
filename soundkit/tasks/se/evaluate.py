"""Evaluate SE task model with given parameters."""
import re
import os
import logging
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import soundfile as sf
import torch
from torchaudio.pipelines import SQUIM_OBJECTIVE
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
from soundkit.defines import SKTaskParams
from soundkit.utils.download_tf_model import build_model, load_model_checkpoint
from soundkit.utils.feature_utils import FeatureExtractor
from soundkit.utils.calculate_feat_stats import feat_stats_estimator
from soundkit.utils.lookaheadBuffer import LookaheadBuffer
from soundkit.utils.tf_stft import tf_istft
from soundkit.utils.tf_complex_utils import polar_to_complex
from soundkit.utils.tf_copy_model import copy_model_weights
from soundkit.utils.basic_dsp import dc_remove
from soundkit.utils.audio import audio_read
from soundkit.utils.plot_api import plot_spectrograms
from soundkit.utils.tf_basic_math import tf_log10_eps

from .datasets import create_raw_tfrecord
from .datasets import create_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    filename="se_evaluate.log",
    )
log = logging.getLogger(__name__)

def evaluate(params: SKTaskParams):
    """Evaluate SE task model with given parameters.

    Args:
        params (SKTaskParams): Task parameters
    """
    logging.info(f"Evaluating SE model with params: {params} and more")
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
            tfrecord, sig, sig)
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

    if params.train["truncate_time"] >= params.data["target_length_in_secs"]:
        raise ValueError(
            f"truncate_time {params.train['truncate_time']} cannot be greater than target_length_in_secs {params.data['target_length_in_secs']}"
        )

    if params.train['truncate_time'] is not None:
        time_steps = int(params.train['truncate_time'] * params.data.signal.sampling_rate //  params.train.feature.hop_size)
    else:
        time_steps = int(params.data['target_length_in_secs'] * params.data.signal.sampling_rate //  params.train.feature.hop_size)

    model_train = build_model(
        params,
        batchsize=batchsize_train,
        dim_feat=dim_feat,
        time_steps=time_steps)

    # load weights from the checkpoint

    load_model_checkpoint(
        model_train, params_evaluate['epoch_loaded'], checkpoint_dir)

    # 3. Compute feature statistics for standardization
    stats = feat_stats_estimator(
        dataset,
        batches,
        folder_nn=checkpoint_dir,
        feat_extractor=feat_extractor,)

    num_fft_bins = feat_params["fft_size"] // 2 + 1
    buffer_sn = LookaheadBuffer(
        num_lookahead=num_lookahead,
        feature_dim=num_fft_bins,
        batchsize=batchsize)

    np_scores = {'sn': np.zeros( (1, 4), dtype=np.float64),
                 'en': np.zeros( (1, 4), dtype=np.float64),}
    stoi_hyp = {'sn': np.zeros( (1, 1), dtype=np.float64),
                 'en': np.zeros( (1, 1), dtype=np.float64),}

    pesq_hyp = {'sn': np.zeros( (1, ), dtype=np.float64),
                 'en': np.zeros( (1,), dtype=np.float64),}
    si_sdr_hyp = {'sn': np.zeros( (1,), dtype=np.float64),
                 'en': np.zeros( (1,), dtype=np.float64),}

    # open a file result.txt to save final results
    result_file_path = os.path.join(result_folder, "result.txt")
    result_file = open(result_file_path, "w")
    
    for step, batch in enumerate(dataset):
        print(f"\rEvaluating (batch) {step}/{batches}, ", end='')

        # Initialize left-over state buffers for streaming STFT
        states_audio_sn = tf.zeros(
            [batchsize, feat_params["frame_size"] - feat_params["hop_size"]],
            dtype=tf.float32
        )

        audio_sn, audio_s, _ = batch

        feat_sn, spec_sn, states_audio_sn = feat_extractor(
            audio_sn, states=states_audio_sn)
        # Apply lookahead
        buffer_sn.reset()
        spec_sn_delay = buffer_sn.apply(spec_sn)

        if params.train['standardization']:
            # Standardize features
            mean_stats = stats['nMean_feat']
            inv_std_stats = stats['nInvStd']
            if params.train['standardization_type'] == 'mean':
                feat_sn_norm = feat_sn - mean_stats
            else:
                feat_sn_norm = (feat_sn - mean_stats) * inv_std_stats
        else:
            feat_sn_norm = feat_sn

        time_steps = feat_sn_norm.shape[1]

        model = build_model(
            params,
            batchsize=batchsize,
            dim_feat=dim_feat,
            time_steps=1500)

        copy_model_weights(
            model_dst=model,
            model_src=model_train)

        if feat_sn_norm.dtype == tf.complex64:
            inputs_nn = tf.stack(
                [tf.math.real(feat_sn_norm),
                    tf.math.imag(feat_sn_norm)],
                axis=-1)
        else:
            inputs_nn = feat_sn_norm
    

        T = inputs_nn.shape[1]
        blks = int(np.ceil(T / 1500))

        tfmask_list = []
        for b in range(blks):
            start = b * 1500
            end = min((b + 1) * 1500, T)
            inputs_nn_blk = inputs_nn[:, start:end]

            rank = tf.rank(inputs_nn_blk)

            if rank == 4:
                inputs_nn_blk = tf.pad(
                inputs_nn_blk,
                paddings=[[0, 0],
                          [0, 1500 - (end - start)],
                          [0, 0],
                          [0, 0]],
                mode='CONSTANT',
                constant_values=0)
            else:
                inputs_nn_blk = tf.pad(
                inputs_nn_blk,
                paddings=[[0, 0],
                          [0, 1500 - (end - start)],
                          [0, 0]],
                mode='CONSTANT',
                constant_values=0)


            tfmask_blk = model(inputs_nn_blk, training=False)
            tfmask_blk = tfmask_blk[:, :end - start, ...]
            tfmask_list.append(tfmask_blk)
        tfmask = tf.concat(tfmask_list, axis=1)

        if feat_sn_norm.dtype == tf.complex64: # complex mask
            if params.train.feature.type == 'erb_complex':
                from soundkit.utils.erb import ERB
                erb = ERB(erb_subband_1=65, erb_subband_2=64)
                tfmask = tf.transpose(tfmask, perm=[0, 3, 1, 2])  # (B,T,F_erb,2) -> (B,2, F_erb,T)
                tfmask = erb.bs(tfmask)
                tfmask = tf.transpose(tfmask, perm=[0, 2, 3, 1])  # (B,2, F_erb,T) -> (B,T,F_erb,2)

            tfmask = tf.complex(
                tfmask[..., 0],
                tfmask[..., 1])
            spec_en_delay = tfmask * spec_sn_delay
        else: # real mask
            pspec_sn_delay = tf.abs(spec_sn_delay)
            phase_sn_delay = tf.math.angle(spec_sn_delay)
            if params.train.feature.type == 'hybrid_mag':
                mat_inv = feat_extractor.mel_filter_inv
                tfmask = tf.matmul(tfmask, mat_inv)
            elif params.train.feature.type == 'erb_mag':
                from soundkit.utils.erb import ERB
                erb = ERB(erb_subband_1=65, erb_subband_2=64)
                tfmask = erb.bs(tfmask)
            pspec_en_delay = tfmask * pspec_sn_delay
            
            spec_en_delay = polar_to_complex(
                pspec_en_delay, phase_sn_delay)

        audio_en = tf_istft(
            spec_en_delay,
            feat_params['frame_size'],
            feat_params['hop_size'],
            feat_params['fft_size'])

        if step < 10:
            # draw spectrograms and tfmask
            name = re.sub(r'(\.wav$|\.flac$)', '.pdf', wavs[step])
            save_path = f"{result_folder}/{name}"
            if feat_sn_norm.dtype == tf.complex64: # complex mask
                pspec_sn_delay = tf.abs(spec_sn_delay)
                pspec_en_delay = tf.abs(spec_en_delay)
                tfmask_real = tf.math.real(tfmask)
                tfmask_imag = tf.math.imag(tfmask)
                rng_mask_real = (
                    tf.reduce_min(tfmask_real).numpy(),
                    tf.reduce_max(tfmask_real).numpy())
                rng_mask_imag = (
                    tf.reduce_min(tfmask_imag).numpy(),
                    tf.reduce_max(tfmask_imag).numpy())
                plot_spectrograms(
                    images=[
                        20 * tf_log10_eps(pspec_sn_delay).numpy()[0].T,
                        tfmask_real.numpy()[0].T,
                        tfmask_imag.numpy()[0].T,
                        20 * tf_log10_eps(pspec_en_delay).numpy()[0].T,
                    ],
                    titles=["Noisy logspec", "TFMask Real", "TFMask Imag", "Enhanced logspec"],
                    vmin_vmax=[(-80, 10), rng_mask_real, rng_mask_imag, (-80, 10)],
                    save_path=save_path,
                )
            else: # real mask
                plot_spectrograms(
                    images=[
                        20 * tf_log10_eps(pspec_sn_delay).numpy()[0].T,
                        tfmask.numpy()[0].T,
                        20 * tf_log10_eps(pspec_en_delay).numpy()[0].T,
                    ],
                    titles=["Noisy logspec", "TFMask", "Enhanced logspec"],
                    vmin_vmax=[(-80, 10), (0, 1), (-80, 10)],
                    save_path=save_path,
                )

            # Save noisy audio
            name = re.sub(r'(\.wav$|\.flac$)', '_sn.wav', wavs[step])
            save_path = f"{result_folder}/{name}"
            audio_sn_np = tf.squeeze(audio_sn, axis=0).numpy()
            sf.write(
                save_path,
                audio_sn_np,
                params.data['signal']['sampling_rate'])
            logging.info(f"Saved noisy audio to {save_path}")

            # save enhanced audio
            name = re.sub(r'(\.wav$|\.flac$)', '_en.wav', wavs[step])
            save_path = f"{result_folder}/{name}"
            audio_en_np = tf.squeeze(audio_en, axis=0).numpy()
            sf.write(
                save_path,
                audio_en_np,
                params.data['signal']['sampling_rate'])
            logging.info(f"Saved enhanced audio to {save_path}")

        objective_model = SQUIM_OBJECTIVE.get_model()

        # stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(torch.from_numpy(audio_sn.numpy()))

        for type_s in ['sn', 'en']:
            if type_s == 'sn':
                audio = audio_sn
            else:
                audio = audio_en

            # Calculate DNSMOS score

            torch_tensor = torch.from_numpy(audio.numpy())
            scores = deep_noise_suppression_mean_opinion_score(
                torch_tensor,
                params.data['signal']['sampling_rate'],
                False)
            print(scores)
            tmp = objective_model(torch.from_numpy(audio.numpy()))
            stoi_hyp[type_s] += tmp[0].detach().numpy()
            pesq_hyp[type_s] += tmp[1].detach().numpy()
            si_sdr_hyp[type_s] += tmp[2].detach().numpy()

            np_scores[type_s] += tf.reduce_sum(scores, axis=0, keepdims=True).numpy()

    metrics = ['STOI', 'PESQ', 'SI-SDR', 'DNSMOS']
    scores = {
        'STOI': stoi_hyp,
        'PESQ': pesq_hyp,
        'SI-SDR': si_sdr_hyp,
        'DNSMOS': np_scores
    }

    # Normalize scores
    for type_s in ['sn', 'en']:
        for key in scores:
            scores[key][type_s] /= (batches * batchsize)

    # Print results
    print("\nAveraged Evaluation Results:")
    
    result_file.write(f"Epoch {params.evaluate.epoch_loaded}: Averaged Evaluation Results:\n")
    for metric in metrics:
        val_sn = scores[metric]['sn']
        val_en = scores[metric]['en']

        if metric == 'DNSMOS':
            result_file.write(f"{metric} Score: noisy {val_sn} | enhanced {val_en} [p808_mos, mos_sig, mos_bak, mos_ovr]\n")
            print(f"{metric} Score: noisy {val_sn} | enhanced {val_en} [p808_mos, mos_sig, mos_bak, mos_ovr]")
        else:
            result_file.write(f"{metric} Score: noisy {val_sn} | enhanced {val_en}\n")
            logging.info(f"{metric} Score: noisy {val_sn} | enhanced {val_en}")