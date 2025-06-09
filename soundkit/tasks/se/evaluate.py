import re
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import torch
import soundfile as sf
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
from .datasets import create_raw_tfrecord
from .datasets import create_dataset
from ...defines import SKTaskParams
from ...utils.download_tf_model import build_model, load_model_checkpoint
from ...utils.feature_utils import FeatureExtractor
from ...utils.calculate_feat_stats import feat_stats_estimator
from ...utils.lookaheadBuffer import LookaheadBuffer
from ...utils.tf_stft import tf_istft
from ...utils.tf_complex_utils import polar_to_complex
from ...utils.tf_copy_model import copy_model_weights
from ...utils.basic_dsp import dc_remove
from ...utils.audio import audio_read
from ...utils.plot_api import plot_spectrograms
from ...utils.tf_basic_math import tf_log10_eps

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
        spec_sn_delay = buffer_sn.apply(spec_sn)

        if params.train['standardization']:
            # Standardize features
            mean_stats = stats['nMean_feat']
            inv_std_stats = stats['nInvStd']
            feat_sn_norm = (feat_sn - mean_stats) * inv_std_stats
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

            tfmask = model(feat_sn_norm, training=False)
        pspec_sn_delay = tf.abs(spec_sn_delay)
        phase_sn_delay = tf.math.angle(spec_sn_delay)

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
            print(f"Saved noisy audio to {save_path}")

            # save enhanced audio
            name = re.sub(r'(\.wav$|\.flac$)', '_en.wav', wavs[step])
            save_path = f"{result_folder}/{name}"
            audio_en_np = tf.squeeze(audio_en, axis=0).numpy()
            sf.write(
                save_path,
                audio_en_np,
                params.data['signal']['sampling_rate'])
            print(f"Saved enhanced audio to {save_path}")
        from torchaudio.pipelines import SQUIM_OBJECTIVE
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
    for metric in metrics:
        val_sn = scores[metric]['sn']
        val_en = scores[metric]['en']
        
        if metric == 'DNSMOS':
            print(f"{metric} Score: noisy {val_sn} | enhanced {val_en} [p808_mos, mos_sig, mos_bak, mos_ovr]")
        else:
            print(f"{metric} Score: noisy {val_sn} | enhanced {val_en}")
    
  