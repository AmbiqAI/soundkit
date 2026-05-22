"""
SoundKit SE Training Script
--------------------------
Organized main training loop for Speech Enhancement (SE) task using TensorFlow/Keras.
Features:
    - Config-driven training
    - Feature extraction and normalization
    - Custom loss functions
    - Model checkpointing and TensorBoard logging
    - Streaming STFT and lookahead buffer support
    - Modular function organization
"""

# === Standard Library Imports ===
import os
import datetime
from pathlib import Path
from typing import Any
import logging
import time

# === Third-Party Imports ===
import tensorflow as tf
from tqdm import tqdm

try:
    import pynvml
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

# === SoundKit Core Imports ===
from soundkit.defines import SKTaskParams
from soundkit.utils.download_tf_model import (
    save_train_log,
    load_train_log,
    build_model,
    get_model_config,
    load_model_checkpoint,
)
from soundkit.utils.feature_utils import FeatureExtractor
from soundkit.utils.losses import LossFactory
from soundkit.utils.calculate_feat_stats import feat_stats_estimator
from soundkit.utils.lookaheadBuffer import LookaheadBuffer
from soundkit.utils.WarmUpCosineDecay import WarmUpCosineDecay
from soundkit.utils.spec_aug import SpecAug
from soundkit.utils.plot_api import (
    plot_spectrograms,
    fig_to_image
)
from soundkit.utils.erb import ERB
from soundkit.utils.tf_basic_math import tf_log10_eps
from soundkit.utils.tf_complex_utils import (
    polar_to_complex,
    complex_magnitude,
    complex_angle,
    get_compressed_complex,
)

from .datasets import create_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
log = logging.getLogger(__name__)


def is_deepfilter_enabled(params: SKTaskParams) -> bool:
    """Return whether DeepFilter is enabled for the current model config."""
    return bool(get_model_config(params).get("is_df", False))

@tf.function
def train_step(
        net: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        loss_fn: Any,
        batch: dict[str, tf.Tensor],
        is_df: bool = False,
        training: bool = True,
        feat_type: str = "mel",
        exp_features :float = 1.0,
        num_lookahead: int = 2,
        erb: ERB = None,
        ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """throughput_
    Executes a single training or validation step for the SE model.

    Args:
        net (tf.keras.Model): The neural network model to train.
        optimizer (tf.keras.optimizers.Optimizer): Optimizer for updating model weights.
        loss_fn (Callable): Loss function to compute training loss.
        batch (dict[str, tf.Tensor]): Batch of input features and targets.
        training (bool): If True, applies gradients and updates weights.

    Returns:
        tuple:
            - loss (tf.Tensor): Computed loss for the batch.
            - est (tf.Tensor): Model output (estimate mask, it is complex valued).
            - spec_en (tf.Tensor): Enhanced spectrogram output.
    """

    feat_sn = batch["feat_sn"]
    lengths = batch["lengths"]

    if feat_sn.dtype == tf.complex64:
        inputs = tf.stack(
            [tf.math.real(feat_sn),
             tf.math.imag(feat_sn)],
            axis=-1)
    else:
        inputs = feat_sn

    if exp_features != 1.0:
        spec_sn = get_compressed_complex(
                batch["spec_sn"],
                exp_features)
        spec_s = get_compressed_complex(
            batch["spec_s"],
            exp_features)
    else:
        spec_sn = batch["spec_sn"]
        spec_s = batch["spec_s"]

    if num_lookahead > 0:
        spec_s_delay = tf.pad(
            spec_s,
            [[0, 0], [num_lookahead, 0], [0, 0]],
            mode='CONSTANT',
            constant_values=0.0)[:, :-num_lookahead]
        spec_sn_delay = tf.pad(
                        spec_sn,
                        [[0, 0], [num_lookahead, 0], [0, 0]],
                        mode='CONSTANT',
                        constant_values=0.0)[:, :-num_lookahead]
    else:
        spec_s_delay = spec_s
        spec_sn_delay = spec_sn

    with tf.GradientTape() as tape:
        est = net(inputs, training=training)
        if feat_sn.dtype == tf.complex64:

            if feat_type == "erb_complex":
                est = tf.transpose(est, perm=[0, 3, 1, 2])  # (B,T,F_erb,2) -> (B,2, F_erb,T)
                est = erb.bs(est)

                est = tf.transpose(est, perm=[0, 2, 3, 1])  # (B,2, F_erb,T) -> (B,T,F_erb,2)

            if not is_df: # no deep filter
                est_real = est[..., 0]
                est_imag = est[..., 1]
                est = tf.complex(
                    est_real,
                    est_imag,
                )

                spec_en_delay = est * spec_sn_delay
                clean = batch["clean"]
            else:
                spec_sn_ext = tf.pad(
                    spec_sn,
                    [[0, 0], [4, 0], [0, 0]],
                    mode='CONSTANT',
                    constant_values=0.0)
                lst = []
                for i in range(5):
                    est_real = est[..., i*2]
                    est_imag = est[..., i*2+1]
                    est1 = tf.complex(
                        est_real,
                        est_imag,
                    )

                    tmp = est1 * spec_sn_ext[:, i:i+est.shape[1]]
                    lst.append(tmp)

                spec_en_delay = tf.reduce_sum(tf.stack(lst, axis=0), axis=0)

                clean = batch["clean"]

                idx_loc=2
                est_real = est[..., 2 * idx_loc]
                est_imag = est[..., 2 * idx_loc + 1]
                est = tf.complex(
                    est_real,
                    est_imag,
                )

            spec_en = spec_en_delay

            loss = loss_fn(
                spec_s_delay,
                spec_en_delay,
                clean=clean,
                lengths=lengths,)
        else:

            if feat_type == "erb_mag":
                est = erb.bs(est[..., 0])
            est = tf.complex(est, 0.0)
            spec_en_delay = est * spec_sn_delay

            spec_s_delay = polar_to_complex(
                complex_magnitude(spec_s_delay),
                complex_angle(spec_s_delay),
            )

            spec_en = spec_en_delay
            loss = loss_fn(
                spec_s_delay,
                spec_en_delay,
                clean=batch["clean"],)

    grad_norms = {}
    if training:
        gradients = tape.gradient(loss, net.trainable_variables)
        gradients_clips = [ tf.clip_by_norm(grad, clip_norm=1.0) if grad is not None else None
                            for grad in gradients ]

        optimizer.apply_gradients(
                    zip(gradients_clips,
                        net.trainable_variables))

        for var, grad in zip(net.trainable_variables, gradients):
            if grad is not None:
                grad_norms[var.name] = tf.norm(grad)
    if exp_features != 1.0:

        spec_en = get_compressed_complex(
            spec_en,
            1.0/exp_features)
    return loss, est, spec_en, grad_norms


def simulate_aec_distortion(audio_sn, audio_s, sr=16000):
    """Simulate AEC-induced speech distortions on mixture and target.

    Both audio_sn (mixture) and audio_s (clean target) receive the SAME
    distortion so the model sees realistic artefacts without an impossible
    learning target.

    Distortions (applied identically to both signals):
      1. Soft clipping  — nonlinear RES artefact (tanh drive).
         A multiplicative mask cannot remove added harmonics, so target
         must also be clipped.
      2. Random segment gain drop — partial near-end cancellation.
         Both signals get the same gain so the model only denoises.

    Args:
        audio_sn: (B, T) float32 noisy mixture tensor.
        audio_s:  (B, T) float32 clean speech tensor (target).
        sr:       sampling rate (default 16000).

    Returns:
        (distorted_sn, distorted_s) — both (B, T) float32.
    """
    B = tf.shape(audio_sn)[0]
    T = tf.shape(audio_sn)[1]

    # ---- 1. Soft clipping (nonlinear AEC artifact) ----
    apply_clip = tf.cast(
        tf.random.uniform([B, 1]) < 0.3, tf.float32)
    drive = tf.random.uniform([B, 1], 2.0, 8.0)

    clipped_sn = tf.math.tanh(audio_sn * drive) / tf.math.tanh(drive)
    audio_sn = audio_sn * (1.0 - apply_clip) + clipped_sn * apply_clip

    clipped_s = tf.math.tanh(audio_s * drive) / tf.math.tanh(drive)
    audio_s = audio_s * (1.0 - apply_clip) + clipped_s * apply_clip

    # ---- 2. Random segment gain reduction (partial cancellation) ----
    for _ in range(3):
        seg_len = tf.random.uniform(
            [], minval=int(0.1 * sr), maxval=int(0.5 * sr), dtype=tf.int32)
        seg_start = tf.random.uniform(
            [], minval=0, maxval=tf.maximum(T - seg_len, 1), dtype=tf.int32)
        indices = tf.range(T)
        in_seg = tf.cast(
            tf.logical_and(indices >= seg_start, indices < seg_start + seg_len),
            tf.float32)                                       # (T,)
        apply_gain = tf.cast(
            tf.random.uniform([B, 1]) < 0.4, tf.float32)
        gain_db = tf.random.uniform([B, 1], -15.0, -3.0)
        gain_linear = tf.pow(10.0, gain_db / 20.0)
        seg_gain = 1.0 - in_seg + in_seg * gain_linear       # (B, T)
        seg_gain = 1.0 - apply_gain + apply_gain * seg_gain
        audio_sn = audio_sn * seg_gain
        audio_s  = audio_s  * seg_gain

    return audio_sn, audio_s

def remix_snr(
        audio_s: tf.Tensor,
        noise: tf.Tensor,
        snr_min_db: float = -5.0,
        snr_max_db: float = 40.0,
        eps: float = 1e-8) -> tuple[tf.Tensor, tf.Tensor]:
    """Remix clean speech and noise to a random SNR per sample.

    Args:
        audio_s: Clean speech tensor of shape (B, T).
        noise: Noise tensor of shape (B, T).
        snr_min_db: Minimum target SNR in dB.
        snr_max_db: Maximum target SNR in dB.
        eps: Small constant for numerical stability.

    Returns:
        Tuple of remixed noisy speech and scaled noise.
    """
    speech_rms = tf.sqrt(tf.reduce_mean(tf.square(audio_s), axis=1, keepdims=True) + eps)
    noise_rms = tf.sqrt(tf.reduce_mean(tf.square(noise), axis=1, keepdims=True) + eps)

    target_snr_db = tf.random.uniform(
        [tf.shape(audio_s)[0], 1],
        minval=snr_min_db,
        maxval=snr_max_db,
        dtype=audio_s.dtype)
    target_ratio = tf.pow(10.0, target_snr_db / 20.0)

    target_noise_rms = speech_rms / target_ratio
    noise_scale = tf.where(
        noise_rms > eps,
        target_noise_rms / noise_rms,
        tf.zeros_like(noise_rms))

    noise_remixed = noise * noise_scale
    audio_sn = audio_s + noise_remixed
    return audio_sn, noise_remixed


def run_epoch(
    config: dict[str, Any],
    dataset: tf.data.Dataset,
    training: bool,
    epoch: int = 0,
) -> tuple[tf.keras.metrics.Mean, tf.keras.metrics.SparseCategoricalAccuracy]:
    """
    Executes one full training or validation epoch over the provided dataset.

    Args:
        config (dict): Contains model, optimizer, loss function, feature extractor, batch counts, and other parameters.
        dataset (tf.data.Dataset): Input data for the epoch.
        training (bool): If True, runs training; if False, runs validation.
        epoch (int): Current epoch number (for logging and scheduling).

    Returns:
        tf.keras.metrics.Mean: Mean loss for the epoch.
    """
    model = config["model"]
    optimizer = config["optimizer"]
    loss_fn = config["loss_fn"]
    params  = config["params"]
    stats = config["feat_stats"]
    is_df = config["is_df"]

    stft_feat = params.train['feature']
    batchsize = params.train["batchsize"]
    feat_extractor = config["feat_extractor"]
    num_lookahead = params.train['num_lookahead']
    total_batches = config["total_batches"]['train'] if training else config["total_batches"]['val']
    aec_distort_prob = getattr(params.train, 'aec_distort_prob', 0.0)
    sr = params.data.signal.sampling_rate

    total_steps=total_batches * epoch

    train_summary_writer = config["train_summary_writer"]
    train_tag = "train" if training else "val"

    loss_metric = tf.keras.metrics.Mean()
    step_time_metric = tf.keras.metrics.Mean()

    # Initialize left-over state buffers for streaming STFT
    num_fft_bins = stft_feat["fft_size"] // 2 + 1

    buffer_sn = LookaheadBuffer(
            num_lookahead=num_lookahead,
            feature_dim=num_fft_bins,
            batchsize=batchsize)
    buffer_s = LookaheadBuffer(
        num_lookahead=num_lookahead,
        feature_dim=num_fft_bins,
        batchsize=batchsize)

    if params.train['spec_aug']:
        specAug_inst = SpecAug(prob=0.3)

    def reset_states():
        states_audio_sn = tf.zeros(
            [batchsize, stft_feat["frame_size"] - stft_feat["hop_size"]],
            dtype=tf.float32
        )
        states_audio_s = tf.zeros_like(states_audio_sn)

        buffer_sn.reset()
        buffer_s.reset()

        model.reset_states()
        return states_audio_sn, states_audio_s

    def random_peak_normalize(
            states_audio_sn,
            min_val=0.1,
            max_val=0.9):
        """
        states_audio_sn: (B, T)
        - If max amplitude > eps: Randomly scale peak to [min_val, max_val].
        - If max amplitude <= eps: Do nothing (scale = 1.0).
        """
        eps = 1e-3

        # 1. Find max amplitude (B, 1)
        max_val = tf.reduce_max(
            tf.abs(states_audio_sn),
            axis=1,
            keepdims=True)

        # 2. Generate random targets (B, 1)
        target_peak = tf.random.uniform(
            tf.shape(max_val), minval=min_val, maxval=max_val, dtype=states_audio_sn.dtype
        )

        # 3. Calculate the scale for the "normal" case
        # We add eps to the denominator to prevent DivisionByZero/Inf in the calculation
        scale_computed = target_peak / (max_val)
        unit_scale = final_scale = tf.where(max_val > eps, 1 / max_val, 1.0)

        # 4. Apply Logic:
        # If max_val > eps  -> Use the computed random scale
        # If max_val <= eps -> Use 1.0 (keep original signal exactly as is)
        final_scale = tf.where(max_val > eps, scale_computed, 1.0)

        return final_scale, unit_scale

    states_audio_sn, states_audio_s = reset_states()
    pbar = tqdm(
        enumerate(dataset),
        total=total_batches,
        desc=f"  [{train_tag}]",
        unit=" batch", ncols=120)
    for step, batch in pbar:
        if params.train['reset_states_every_batch']:
            # Random reset: 30% hard reset (cold start), 70% carry-over (streaming)
            # if training and tf.random.uniform([]) > 0.3:
            #     pass  # keep states from previous batch
            # else:
            #     states_audio_sn, states_audio_s = reset_states()
            states_audio_sn, states_audio_s = reset_states()
        audio_sn, audio_s, lengths = batch

        # deep copy of clean for VAD computation
        clean = tf.identity(audio_s)
        
        # # remix snr
        # if training:
        #     noise = audio_sn - audio_s
        #     audio_sn, noise = remix_snr(
        #         audio_s,
        #         noise,
        #         snr_min_db=-10.0,
        #         snr_max_db=40.0)

        if 1: # add some residue noise to the target
            noise = audio_sn - audio_s

            # AEC distortion: distort mixture and target with the same artefacts
            if training and aec_distort_prob > 0:
                if tf.random.uniform([]) < aec_distort_prob:
                    audio_sn, audio_s = simulate_aec_distortion(audio_sn, audio_s, sr=sr)
                    clean = audio_s
                    noise = audio_sn - audio_s

            ns_db = tf.random.uniform(
                [audio_sn.shape[0], 1],
                -50.0, -45.0)
            factor = tf.pow(10.0, ns_db / 20.0)
            audio_s = audio_s + noise * 0
        if 1:
            gain, unit_gain = random_peak_normalize(
                audio_sn,
                min_val = params.data.min_amp,
                max_val = params.data.max_amp)

            audio_sn = audio_sn * gain
            audio_s = audio_s * gain
            clean = clean * gain
        step_start = time.perf_counter()

        # Extract features using streaming state
        feat_sn, spec_sn, states_audio_sn = feat_extractor(
            audio_sn, states=states_audio_sn)
        _, spec_s, states_audio_s = feat_extractor(
            audio_s, states=states_audio_s)
        # if params.train.feature.exp_complex != 1.0:
        #     spec_sn = get_compressed_complex(
        #         spec_sn,
        #         params.train.feature.exp_complex,
        #         params.train.feature.eps)
        #     spec_s = get_compressed_complex(
        #         spec_s,
        #         params.train.feature.exp_complex,
        #         params.train.feature.eps)
        if params.train['spec_aug'] and training:
            feat_sn, _ = specAug_inst(
                feat_sn,
                feat_sn   # dummy y — only feat_sn matters
            )


        # mag = tf.abs(feat_sn)
        # scale = (mag**0.5) / (mag + 2**-15)  # All float32 math
        # feat_sn = feat_sn * tf.cast(scale, tf.complex64) # Cast and apply
        if params.train['standardization']:
            # Standardize features
            if params.train['standardization_type'] in ["mve", "mean", "std"]:
                mean_stats = stats['nMean_feat']
                inv_std_stats = stats['nInvStd']
                feat_sn_norm = (feat_sn - mean_stats) * tf.complex(inv_std_stats, 0.0)
            elif params.train['standardization_type'] == "constant":
                feat_sn_norm = feat_sn / 32
        else:
            # No standardization, use raw features
            feat_sn_norm = feat_sn


        batch_data = {
            "spec_sn": spec_sn,
            "spec_s":  spec_s,
            "clean": clean,
            "feat_sn": feat_sn_norm,
            "mask": 1.0,
            "hop_size": stft_feat['hop_size'],
            "frame_size": stft_feat['frame_size'],
            "fft_size": stft_feat['fft_size'],
            "lengths": lengths,
        }

        loss, logits, spec_en, grad_norms = train_step(
            model,
            optimizer,
            loss_fn,
            batch_data,
            is_df=is_df,
            feat_type=stft_feat['type'],
            training=training,
            exp_features = params.train.feature.exp_complex,
            num_lookahead = num_lookahead,
            erb = getattr(feat_extractor, 'erb', None),
            )

        loss_metric.update_state(loss)

        # Gradient flow diagnostics: detect vanishing/exploding gradients
        if training and grad_norms and step % 50 == 0:
            vanishing_layers = []
            exploding_layers = []
            healthy_min, healthy_max = 1e-7, 1e3
            total_norm = 0.0
            for var_name, norm_val in grad_norms.items():
                nv = float(norm_val.numpy())
                total_norm += nv ** 2
                if nv < healthy_min:
                    vanishing_layers.append((var_name, nv))
                elif nv > healthy_max:
                    exploding_layers.append((var_name, nv))
            total_norm = total_norm ** 0.5

            if vanishing_layers or exploding_layers:
                log.warning(f"[Step {step}] Gradient flow issues detected!")
                if vanishing_layers:
                    log.warning(f"  VANISHING ({len(vanishing_layers)} layers):")
                    for name, nv in vanishing_layers[:5]:
                        log.warning(f"    {name}: {nv:.2e}")
                if exploding_layers:
                    log.warning(f"  EXPLODING ({len(exploding_layers)} layers):")
                    for name, nv in exploding_layers[:5]:
                        log.warning(f"    {name}: {nv:.2e}")

            if train_summary_writer is not None:
                with train_summary_writer.as_default():
                    tf.summary.scalar('grad_flow/total_norm', total_norm, step=total_steps)
                    tf.summary.scalar('grad_flow/num_vanishing', len(vanishing_layers), step=total_steps)
                    tf.summary.scalar('grad_flow/num_exploding', len(exploding_layers), step=total_steps)
                    for var_name, norm_val in grad_norms.items():
                        tf.summary.scalar(f'grad_norm/{var_name}', norm_val, step=total_steps)
        # acc_metric.update_state(y_batch, logits)  # accuracy not computed yet

        if training:
            # Log training metrics
            if train_summary_writer is not None:
                with train_summary_writer.as_default():
                    tf.summary.scalar(
                        'learning_rate',
                        optimizer.learning_rate,
                        step=total_steps)

        # Running average step time (ms)
        step_time_ms = (time.perf_counter() - step_start) * 1000.0
        step_time_metric.update_state(step_time_ms)
        if train_summary_writer is not None:
            with train_summary_writer.as_default():
                tf.summary.scalar(f'{train_tag}/step_time_ms', step_time_metric.result(), step=total_steps)

        # GPU metrics every 10 steps (if NVML available)
        if _NVML_AVAILABLE and (step % 10 == 0) and train_summary_writer is not None:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                with train_summary_writer.as_default():
                    for idx in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        tf.summary.scalar(f'gpu/{idx}/utilization_percent', util.gpu, step=total_steps)
                        mem_used_pct = (mem.used / max(mem.total, 1)) * 100.0
                        tf.summary.scalar(f'gpu/{idx}/memory_used_percent', mem_used_pct, step=total_steps)
            except Exception:
                pass

        total_steps += 1
        # Update progress bar
        pbar.set_postfix(loss=f"{loss_metric.result():.5f}", step_ms=f"{step_time_metric.result():.1f}")

        if step % 100 == 0:
 
            spec_en = tf.abs(spec_en)
            pspec_en = 20*tf_log10_eps( tf.abs(spec_en[0])).numpy()

            if feat_sn_norm.dtype == tf.complex64:
                logits_real = tf.math.real(logits)
                logits_imag = tf.math.imag(logits)
                mask_real = logits_real[0].numpy()
                mask_imag = logits_imag[0].numpy()
                mask_real_range=(mask_real.min(), mask_real.max())
                mask_imag_range=(mask_imag.min(), mask_imag.max())
            else:
                logits = tf.abs(logits)
                mask_range=(
                    logits[0].numpy().min(), logits[0].numpy().max())
                mask = logits[0].numpy()

            pspec_sn = 20*tf_log10_eps( tf.abs(spec_sn[0])).numpy()
            pspec_s = 20*tf_log10_eps( tf.abs(spec_s[0])).numpy()

            if params.train['feature']['type'] in ('mel', 'logpspec', 'hybrid'):
                feat_sn = 10* feat_sn[0].numpy()
                feat_sn_norm_d = 10* feat_sn_norm[0].numpy()
            elif params.train['feature']['type'] in ('pspec', 'spec', "hybrid_mag", "erb_mag"):
                feat_sn = 20*tf_log10_eps( tf.abs(feat_sn[0])).numpy()
                feat_sn_norm_d = 20*tf_log10_eps( tf.abs(feat_sn_norm[0])).numpy()
            elif params.train['feature']['type'] in ("erb_complex"):
                feat_sn_norm_d = tf.abs(feat_sn_norm[0]).numpy()**params.train['feature']['exp_complex']
            if feat_sn_norm.dtype == tf.complex64:
                fig, axes = plot_spectrograms(
                    images=[pspec_s.T, pspec_sn.T, feat_sn_norm_d.T, pspec_en.T, mask_real.T, mask_imag.T],
                    titles=["clean logspec", "noisy logspec", "feat", "enhanced logspec", "mask real", "mask imag"],
                    vmin_vmax=[(-80, 10), (-80, 10), (feat_sn_norm_d.min(), feat_sn_norm_d.max()), (-80, 10), mask_real_range, mask_imag_range],
                    show_colorbar=True,
                    show_fig=False       # set False if only saving
                )
                
            else:
                fig, axes = plot_spectrograms(
                    images=[pspec_s.T, pspec_sn.T, feat_sn.T, pspec_en.T, mask.T],
                    titles=["clean logspec", "noisy logspec", "feat", "enhanced logspec", "mask"],
                    vmin_vmax=[(-80, 10), (-80, 10), (-80, 10), (-80, 10), mask_range],
                    show_colorbar=True,
                    show_fig=False       # set False if only saving
                )

            valid_frames = int((lengths[0] // stft_feat['hop_size']).numpy())
            for ax in axes:
                num_frames = ax.images[0].get_array().shape[1]
                valid_x = min(max(valid_frames, 0), num_frames)
                ax.axvline(valid_x - 0.5, color="cyan", linewidth=1.2)
                if valid_x < num_frames:
                    ax.axvspan(valid_x - 0.5, num_frames - 0.5, color="black", alpha=0.18)

            # Convert fig to image
            tf_image = fig_to_image(fig)

            # Write to TensorBoard
            if train_summary_writer is not None:
                with train_summary_writer.as_default():
                    tf.summary.image("spectrograms", tf_image, step=epoch)
    # Final summary for the epoch
    pbar.close()

    return loss_metric

def train(params: SKTaskParams):
    """Train beat task model with given parameters.

    Args:
        params (SKTaskParams): Task parameters
    """
    print(f"Training SE model with params: {params} and more")

    params_train = params.train
    checkpoint_dir = f"{params.train['path']['checkpoint_dir']}"
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tfboard_dir = f"{params.train['path']['tensorboard_dir']}/logs/{current_time}"
    train_summary_writer = tf.summary.create_file_writer(tfboard_dir)
    batchsize = params.train['batchsize']
    if batchsize < 8:
        log.warning(
            "Batch size is very small. This may lead to slow training and unstable gradients. "
            "Consider increasing the batch size in your config.yaml file for better performance. "
            "Or set it to a value that fits your GPU memory. "
            "Or choose the highest value that fits within your available memory."
        )

    # 1. Define feature extractor
    feat_extractor = FeatureExtractor(
        params=params,
    )
    dim_feat = feat_extractor.dim_feat

    # 2. Build the model

    # Load from YAML file

    if params.train["truncate_time"] > params.data["target_length_in_secs"]:
        raise ValueError(
            f"truncate_time {params.train['truncate_time']} cannot be greater than target_length_in_secs {params.data['target_length_in_secs']}"
        )

    if params.train['truncate_time'] is not None:
        timesteps = int(params.train['truncate_time'] * params.data.signal.sampling_rate //  params.train.feature.hop_size)
    else:
        timesteps = int(params.data['target_length_in_secs'] * params.data.signal.sampling_rate //  params.train.feature.hop_size)

    model = build_model(
        params,
        batchsize,
        dim_feat,
        time_steps=timesteps)

    _, epoch_loaded_1 = load_model_checkpoint(
        model, params_train['epoch_loaded'], checkpoint_dir)

    # 3. Create the dataset
    tfrecord_list = {
        'train': Path(params.data['path_tfrecord']) / params.data['tfrecord_datalist_name']['train'],
        'val':  Path(params.data['path_tfrecord']) / params.data['tfrecord_datalist_name']['val'],
    }
    truncate_samples = int(params.train['truncate_time'] * params.data.signal.sampling_rate) if params.train['truncate_time'] is not None else None

    # if params.train.num_per_epoch_files.train

    ds_train, batches_train = create_dataset(
        tfrecord_list['train'],
        batchsize=batchsize,
        is_shuffle=True,
        num_per_epoch_files=params.train.num_per_epoch_files.train,
        truncate_samples=truncate_samples,
    )
    ds_val, batches_val = create_dataset(
        tfrecord_list['val'],
        batchsize=batchsize,
        is_shuffle=False,
        num_per_epoch_files=params.train.num_per_epoch_files.val,
        truncate_samples=truncate_samples,
    )

    # 4. Compute feature statistics for standardization
    if params_train['standardization']:
        stats = feat_stats_estimator(
            ds_train,
            batches_train,
            folder_nn=checkpoint_dir,
            feat_extractor=feat_extractor,
            standardization_type=params_train['standardization_type'],)
    else:
        stats = {
            'nMean_feat': tf.zeros([dim_feat], dtype=tf.float32),
            'nInvStd': tf.ones([dim_feat], dtype=tf.float32),
        }

    # 5. Define loss function
    loss_fn = LossFactory.get(
        params.train["loss_function"]["type"],
        params=params.train["loss_function"]["params"])

    if params_train['lr_schedule'] == "cosine":
        lr_schedule = WarmUpCosineDecay(
            initial_lr = float(params_train['initial_lr']),
            total_steps = params_train['epochs'] * batches_train,
            warmup_steps =params_train['warmup_epochs'] * batches_train,
            alpha=1e-5,
            initial_step=epoch_loaded_1 * batches_train,)
    elif params_train['lr_schedule'] == "constant":
        lr_schedule = params_train['initial_lr']
    else:
        raise ValueError(f"Unknown lr_schedule: {params_train['lr_schedule']}")

    # 6. Define optimizer
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-5,
        beta_1=0.9,
        beta_2=0.98,
        )
    is_df = is_deepfilter_enabled(params)

    # 7. Training loop
    # Load previous log if it exists

    log_path = f"{checkpoint_dir}/train_log.json"
    train_log = load_train_log(log_path, params_train['epochs'])
    # import pdb; pdb.set_trace()
    for epoch in range(epoch_loaded_1, params_train['epochs']):
        log_epoch ={"epoch": epoch}
        train_config={
            'params': params,
            'feat_stats':  stats,
            'feat_extractor': feat_extractor,
            'model': model,
            'optimizer': optimizer,
            'loss_fn': loss_fn,
            'is_df': is_df,
            'total_batches': {
                'train': batches_train,
                'val': batches_val,
                },
            'train_summary_writer': train_summary_writer,
            }
        print(f"Epoch {epoch}/{params_train['epochs']}\n")

        # Training phase
        if not params.train['debug']:
            loss = run_epoch(
                train_config,
                ds_train,
                training=True,
                epoch=epoch,
            )
            log_epoch["train_loss"] = float(loss.result().numpy())
            if train_summary_writer is not None:
                with train_summary_writer.as_default():
                    tf.summary.scalar(f"train/loss", loss.result(), step=epoch)

        # Validation phase
        loss = run_epoch(
            train_config,
            ds_val,
            training=False,
            epoch=epoch,
        )
        log_epoch["val_loss"] = float(loss.result().numpy())
        if train_summary_writer is not None:
            with train_summary_writer.as_default():
                tf.summary.scalar(f"val/loss", loss.result(), step=epoch)

        try:
            train_log[epoch] = log_epoch
        except:
            train_log.append(log_epoch)

        save_train_log(train_log, log_path)

        # 5. Save model weights
        os.makedirs("checkpoints", exist_ok=True)

        model.save_weights(f"{checkpoint_dir}/checkpoints/model_checkpoint_ep{epoch}")
