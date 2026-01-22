"""
soundKIT SE Training Script
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
try:
    import pynvml
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

# === soundKIT Core Imports ===
from soundkit.defines import SKTaskParams
from soundkit.utils.download_tf_model import (
    save_train_log,
    load_train_log,
    build_model,
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
)

from .datasets import create_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
log = logging.getLogger(__name__)

erb = ERB(erb_subband_1=65, erb_subband_2=64)

@tf.function
def train_step(
        net: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        loss_fn: Any,
        batch: dict[str, tf.Tensor],
        training: bool = True,
        feat_type: str = "mel",
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

    if feat_sn.dtype == tf.complex64:
        inputs = tf.stack(
            [tf.math.real(feat_sn),
             tf.math.imag(feat_sn)],
            axis=-1)
    else:
        inputs = feat_sn
    with tf.GradientTape() as tape:
        est = net(inputs, training=training)
        if feat_sn.dtype == tf.complex64:

            if feat_type == "erb_complex":
                est = tf.transpose(est, perm=[0, 3, 1, 2])  # (B,T,F_erb,2) -> (B,2, F_erb,T)
                est = erb.bs(est)
                est = tf.transpose(est, perm=[0, 2, 3, 1])  # (B,2, F_erb,T) -> (B,T,F_erb,2)
            est_real = est[..., 0]
            est_imag = est[..., 1]
            est = tf.complex(
                est_real,
                est_imag,
            )
            spec_en_delay = est * batch["spec_sn_delay"]
            spec_en = spec_en_delay

            loss = loss_fn(batch["spec_s_delay"], spec_en_delay )
        else:
            if feat_type == "erb_mag":
                est = erb.bs(est[..., 0])
            est = tf.complex(est, 0.0)
            spec_en_delay = est * batch["spec_sn_delay"]

            spec_s_delay = polar_to_complex(
                complex_magnitude(batch["spec_s_delay"]),
                complex_angle(batch["spec_sn_delay"]),
            )

            spec_en = spec_en_delay
            loss = loss_fn(spec_s_delay, spec_en_delay)

    if training:
        gradients = tape.gradient(loss, net.trainable_variables)
        gradients_clips = [ tf.clip_by_norm(grad, clip_norm=1.0) if grad is not None else None
                            for grad in gradients ]

        optimizer.apply_gradients(
                    zip(gradients_clips,
                        net.trainable_variables))

    return loss, est, spec_en

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

    stft_feat = params.train['feature']
    batchsize = params.train["batchsize"]
    feat_extractor = config["feat_extractor"]
    num_lookahead = params.train['num_lookahead']
    total_batches = config["total_batches"]['train'] if training else config["total_batches"]['val']

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
        specAug_inst = SpecAug()

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
        eps = 1e-9

        # 1. Find max amplitude (B, 1)
        max_val = tf.reduce_max(tf.abs(states_audio_sn), axis=1, keepdims=True)

        # 2. Generate random targets (B, 1)
        target_peak = tf.random.uniform(
            tf.shape(max_val), minval=min_val, maxval=max_val, dtype=states_audio_sn.dtype
        )

        # 3. Calculate the scale for the "normal" case
        # We add eps to the denominator to prevent DivisionByZero/Inf in the calculation
        scale_computed = target_peak / (max_val + eps)

        # 4. Apply Logic:
        # If max_val > eps  -> Use the computed random scale
        # If max_val <= eps -> Use 1.0 (keep original signal exactly as is)
        final_scale = tf.where(max_val > eps, scale_computed, 1.0)

        return final_scale
    states_audio_sn, states_audio_s = reset_states()
    for step, batch in enumerate(dataset):
        if params.train['reset_states_every_batch']:
            rn = tf.random.uniform([], 0, 1.0)
            if rn < 0.5:
                states_audio_sn, states_audio_s = reset_states()

        audio_sn, audio_s, _ = batch
        gain = random_peak_normalize(
            states_audio_sn,
            min_val = params.data.min_amp,
            max_val = params.data.max_amp)

        audio_sn = audio_sn * gain
        audio_s = audio_s * gain

        step_start = time.perf_counter()

        # Extract features using streaming state
        feat_sn, spec_sn, states_audio_sn = feat_extractor(
            audio_sn, states=states_audio_sn)
        _, spec_s, states_audio_s = feat_extractor(
            audio_s, states=states_audio_s)

        # Apply lookahead
        spec_sn_delay = buffer_sn.apply(spec_sn)
        spec_s_delay = buffer_s.apply(spec_s)

        if params.train['spec_aug'] and training:
            feat_sn = specAug_inst(
                feat_sn,
            )

        if params.train['standardization']:
            # Standardize features

            mean_stats = stats['nMean_feat']
            inv_std_stats = stats['nInvStd']
            if params.train['standardization_type'] == "mean":
                feat_sn_norm = feat_sn - mean_stats
            else: # "mve"
                feat_sn_norm = (feat_sn - mean_stats) * inv_std_stats
        else:
            # No standardization, use raw features
            feat_sn_norm = feat_sn

        batch_data = {
            "spec_sn_delay": spec_sn_delay,
            "spec_s_delay": spec_s_delay,
            "feat_sn": feat_sn_norm,
            "mask": 1.0
        }

        loss, logits, spec_en = train_step(
            model,
            optimizer,
            loss_fn,
            batch_data,
            feat_type=stft_feat['type'],
            training=training,
            )

        loss_metric.update_state(loss)
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
        # Print inline batch progress
        print(
            f"  [{train_tag}] | {step + 1}/{total_batches} | "
            f"    Loss: {loss_metric.result()} | ",
            end="\r",
            flush=True
        )

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
            elif params.train['feature']['type'] in ('pspec', 'spec', "erb_complex", "hybrid_mag", "erb_mag"):
                feat_sn = 20*tf_log10_eps( tf.abs(feat_sn[0])).numpy()
            if feat_sn_norm.dtype == tf.complex64:
                fig, axes = plot_spectrograms(
                    images=[pspec_s.T, pspec_sn.T, feat_sn.T, pspec_en.T, mask_real.T, mask_imag.T],
                    titles=["clean logspec", "noisy logspec", "feat", "enhanced logspec", "mask real", "mask imag"],
                    vmin_vmax=[(-80, 10), (-80, 10), (-80, 10), (-80, 10), mask_real_range, mask_imag_range],
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

            # Convert fig to image
            tf_image = fig_to_image(fig)

            # Write to TensorBoard
            if train_summary_writer is not None:
                with train_summary_writer.as_default():
                    tf.summary.image("spectrograms", tf_image, step=epoch)
    # Final summary for the epoch
    print(
        f"  [{train_tag}] |\n"
    )

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

    if params.train["truncate_time"] >= params.data["target_length_in_secs"]:
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

    # if params.train.num_per_epoch_files.train >

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
            feat_extractor=feat_extractor,)
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
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        weight_decay=0.1
        )

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
