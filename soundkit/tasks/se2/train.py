import os
import datetime
from pathlib import Path
from typing import Any
import tensorflow as tf
import soundfile as sf
import numpy as np
from soundkit.defines import SKTaskParams
from soundkit.utils.download_tf_model import save_train_log, load_train_log
from soundkit.utils.download_tf_model import build_model, load_model_checkpoint
from soundkit.utils.feature_utils import FeatureExtractor
from soundkit.utils.losses import LossFactory
from soundkit.utils.calculate_feat_stats import feat_stats_estimator, mean_varinace_norm
from soundkit.utils.lookaheadBuffer import LookaheadBuffer
from soundkit.utils.WarmUpCosineDecay import WarmUpCosineDecay
from soundkit.utils.plot_api import plot_spectrograms
from soundkit.utils.tf_basic_math import tf_log10_eps
from soundkit.utils.tf_complex_utils import complex_to_polar, polar_to_complex, realarray_to_complex
from .datasets import create_dataset
pi = np.pi  # Use NumPy's constant
SAVE_FIG = True  # Set to True for time-domain training, False for frequency-domain  

import tensorflow as tf

def pre_emphasis(sig, coeff=0.97):
    """
    Apply pre-emphasis filter to a batch of signals.
    
    Args:
        sig: Tensor of shape (B, T)
        coeff: Pre-emphasis coefficient (typically 0.95 to 0.97)

    Returns:
        Tensor of shape (B, T), pre-emphasized signals
    """
    # Pad first value of each signal with itself for correct shape
    padded = tf.pad(sig[:, :-1], [[0, 0], [1, 0]], mode='CONSTANT')

    # Apply the filter: y[n] = x[n] - α * x[n-1]
    emphasized = sig - coeff * padded

    return emphasized

import tensorflow as tf

def de_emphasis(sig, coeff=0.97):
    """
    Apply de-emphasis (inverse of pre-emphasis) to a batch of signals.
    
    Args:
        sig: Tensor of shape (B, T), pre-emphasized signals
        coeff: Pre-emphasis coefficient (same used in pre-emphasis)

    Returns:
        Tensor of shape (B, T), de-emphasized signals
    """
    def de_emphasize_single(x):
        return tf.scan(lambda a, y: y + coeff * a, x)

    return tf.map_fn(de_emphasize_single, sig)


@tf.function
def train_step(
        net: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        loss_fn: Any,
        batch: dict[str, tf.Tensor],
        training: bool = True,
        loss_type: str = "mse",
        overlap: int = 320,
        feat_extractor: Any = None,
        fftsize: int = 512):
    """Perform a single training step."""

    feat_sn = batch["feat_sn"]

    with tf.GradientTape() as tape:
        est_signal_frames = net(feat_sn, training=training)

        est_signal = feat_extractor.reconstruct(
            est_signal_frames)

        # est_signal = de_emphasis(est_signal)

        est_signal = tf.clip_by_value(est_signal, -1.0, 1.0)

        clean_signal = feat_extractor.reconstruct(
            batch["input_s_delay"])

        # clean_signal = de_emphasis(clean_signal)

        est_signal   =   est_signal[:, overlap:-overlap]
        clean_signal = clean_signal[:, overlap:-overlap]

        if 0:
            loss = tf.reduce_mean(

                    tf.sqrt((est_signal - clean_signal)**2 + 1e-12)

                )
            # loss = si_sdr_loss(est_signal, clean_signal)

        else:
            spec_s = tf.signal.rfft(batch["input_s_delay"], [fftsize])
            spec_en = tf.signal.rfft(est_signal_frames, [fftsize])
            # spec_sn = tf.signal.rfft(batch["input_sn_delay"], [fftsize])

            # mag_en, _ = complex_to_polar(spec_en)
            # _, phase_sn = complex_to_polar(spec_sn, eps=0)
            # spec_en = polar_to_complex(mag_en, phase_sn)


            loss = loss_fn(
                y_true=spec_s,
                y_pred=spec_en)


    if training:
        gradients = tape.gradient(loss, net.trainable_variables)
        gradients_clips = [ tf.clip_by_norm(grad, clip_norm=1.0)
                            for grad in gradients ]

        optimizer.apply_gradients(
                    zip(gradients_clips,
                        net.trainable_variables))

    return loss, clean_signal, est_signal, est_signal_frames


def run_epoch(
    config: dict[str, Any],
    dataset: tf.data.Dataset,
    training: bool,
    epoch: int = 0,
) -> tuple[tf.keras.metrics.Mean, tf.keras.metrics.SparseCategoricalAccuracy]:
    """
    Run a single training or evaluation epoch.

    Args:
        config (dict): Configuration dictionary containing:
            - model: tf.keras.Model
            - optimizer: tf.keras.optimizers.Optimizer
            - loss_fn: Callable loss function
            - signal: dict with 'frame_size' and 'hop_size'
            - batchsize: int
            - feat_extractor: callable feature extraction module
            - total_batches: dict with 'train' and 'val' counts
        dataset (tf.data.Dataset): The dataset to iterate over.
        training (bool): Whether this is a training epoch (True) or validation (False).

    Returns:
        tuple:
            - tf.keras.metrics.Mean: Average loss across the epoch
            - tf.keras.metrics.SparseCategoricalAccuracy: Accuracy metric (unused here)
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

    # Initialize left-over state buffers for streaming STFT
    states_audio_sn = tf.zeros(
        [batchsize, stft_feat["frame_size"] - stft_feat["hop_size"]],
        dtype=tf.float32
    )

    states_audio_s = tf.zeros(
        [batchsize, stft_feat["frame_size"] - stft_feat["hop_size"]],
        dtype=tf.float32
    )


    num_fft_bins = stft_feat["frame_size"]
    dtype = tf.float32


    buffer_sn = LookaheadBuffer(
        num_lookahead=num_lookahead,
        feature_dim=num_fft_bins,
        batchsize=batchsize,
        dtype=dtype)

    buffer_s = LookaheadBuffer(
        num_lookahead=num_lookahead,
        feature_dim=num_fft_bins,
        batchsize=batchsize,
        dtype=dtype)

    from test_tfsignal import FrameClass

    frame_class = FrameClass(
        frame_length=stft_feat["frame_size"],
        frame_step=stft_feat["hop_size"],
    )

    for step, batch in enumerate(dataset):
        audio_sn, audio_s, _ = batch

        # audio_sn = pre_emphasis(audio_sn)
        # audio_s = pre_emphasis(audio_s)
        if params.train['reset_every_batch']:
            model.reset_states()

        # Extract features using streaming state

        frame_class.reset_states()
        feat_sn = frame_class.apply_frames(
            audio_sn)

        frame_class.reset_states()
        feat_s = frame_class.apply_frames(
            audio_s)

        # Apply lookahead
        input_sn_delay = buffer_sn.apply(feat_sn)
        input_s_delay = buffer_s.apply(feat_s)
        if params.train['standardization']:
            # Standardize features

            mean_stats = stats['nMean_feat']
            inv_std_stats = stats['nInvStd']
            feat_sn_norm = mean_varinace_norm(
                feat_sn,
                mean_stats=mean_stats,
                inv_std_stats=inv_std_stats)
        else:
            # No standardization, use raw features
            feat_sn_norm = feat_sn

        batch_data = {
            "input_sn_delay": input_sn_delay,
            "input_s_delay": input_s_delay,
            "feat_sn": feat_sn_norm,
            "mask": 1.0
        }

        # Perform one training or val step
        loss, clean_signal, est_signal, est_signal_frames = train_step(
            model,
            optimizer,
            loss_fn,
            batch_data,
            training=training,
            loss_type=params.train['loss_function']['type'],
            overlap=stft_feat["frame_size"] - stft_feat["hop_size"],
            feat_extractor=frame_class,
            fftsize=stft_feat["fft_size"],)
        # print(f"max: {tf.reduce_max(est_signal)} min: {tf.reduce_min(est_signal)}")
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

        total_steps += 1
        # Print inline batch progress
        print(
            f"  [{train_tag}] | {step + 1}/{total_batches} | "
            f"    Loss: {loss_metric.result()} | ",
            end="\r",
            flush=True
        )

        if 1:

            if step % 100 == 0:
                idx_image = 0

                sig_sn = frame_class.reconstruct(input_sn_delay)
                # sig_sn = de_emphasis(sig_sn)

                sig_s = frame_class.reconstruct(input_s_delay)
                # sig_s = de_emphasis(sig_s)

                sig_en = frame_class.reconstruct(est_signal_frames)
                # sig_en = de_emphasis(sig_en)

                frs_sig_sn = frame_class.apply_frames(sig_sn)
                frs_sig_s = frame_class.apply_frames(sig_s)
                frs_sig_en = frame_class.apply_frames(sig_en)

                spec_sn = tf.signal.rfft(frs_sig_sn, [stft_feat["fft_size"]])
                spec_s  = tf.signal.rfft(frs_sig_s, [stft_feat["fft_size"]])
                spec_en = tf.signal.rfft(frs_sig_en, [stft_feat["fft_size"]])

                type_emph='log'
                if type_emph == 'log':
                    emphasis_func = lambda x: 20 * tf_log10_eps(tf.abs(x)).numpy()
                    limits=(-80, 10)

                elif type_emph == 'comp_exp':
                    emphasis_func = lambda x: (tf.abs(x + 1e-8) ** 0.6).numpy()
                    limits=(0, 0.5)
                elif type_emph == 'linear':
                    emphasis_func = lambda x: tf.abs(x).numpy()

                pspec_sn = emphasis_func(spec_sn[idx_image])
                pspec_s  = emphasis_func(spec_s[idx_image])

                feat_sn  = emphasis_func(feat_sn[idx_image])

                pspec_en = emphasis_func(spec_en[idx_image])

                
                if SAVE_FIG:
                    # import pdb; pdb.set_trace()  # Debugging breakpoint
                    fig = plot_spectrograms(
                        images=[pspec_s.T, pspec_sn.T, pspec_en.T],
                        titles=["clean logspec", "noisy logspec", "enhanced logspec"],
                        # vmin_vmax=[(-80, 10), (-80, 10), (-80, 10)],
                        vmin_vmax=[limits, limits, limits],
                        
                        # vmin_vmax=[(pspec_s.min(), pspec_s.max()), (pspec_sn.min(), pspec_sn.max()), (pspec_en.min(), pspec_en.max())],
                        show_colorbar=True,
                        show_fig=False       # set False if only saving
                    )

                    from ...utils.plot_api import fig_to_image
                    # Convert fig to image
                    tf_image = fig_to_image(fig)

                    # Write to TensorBoard
                    if train_summary_writer is not None:
                        with train_summary_writer.as_default():
                            tf.summary.image("spectrograms", tf_image, step=epoch)

                    sig = np.stack([sig_sn[idx_image].numpy(), sig_en[idx_image].numpy()], axis=-1)
                    # import pdb; pdb.set_trace()  # Debugging breakpoint
                    sf.write("test.wav", sig, params.data.signal.sampling_rate)
                    sf.write("test_en.wav", sig_en[idx_image].numpy(), params.data.signal.sampling_rate)
                    sf.write("test_s.wav", sig_s[idx_image].numpy(), params.data.signal.sampling_rate)
                    sf.write("test_sn.wav", sig_sn[idx_image].numpy(), params.data.signal.sampling_rate)

                    # import sounddevice as sd
                    # sd.play(sig_en, samplerate=params.data.signal.sampling_rate)
                    # import pdb; pdb.set_trace()  # Debugging breakpoint
                    # sd.play(sig, samplerate=params.data.signal.sampling_rate)   
                    
                    # sf.write()
                    # import pdb; pdb.set_trace()  # Debugging breakpoint

    # Final summary for the epoch
    print(
        f"  [{train_tag}] |\n"
    )

    return loss_metric

def train(params: SKTaskParams):
    """Train beat task model with given parameters.

    Args:
        params (HKTaskParams): Task parameters
    """
    print(f"Training SE model with params: {params} and more")

    params_train = params.train
    checkpoint_dir = f"{params.train['path']['checkpoint_dir']}"
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tfboard_dir = f"{params.train['path']['tensorboard_dir']}/logs/{current_time}"
    train_summary_writer = tf.summary.create_file_writer(tfboard_dir)
    batchsize = params.train['batchsize']
    if batchsize < 8:
        print("WARNING: Batch size is very small, consider increasing it for better performance. Setting \n\n" \
        "in your config file:\n\n" \
        "train:\n" \
        "  batchsize: 32\n\n" \
        "or higher.\n\n")
    # 1. Define feature extractor
    feat_extractor = FeatureExtractor(
        params=params,
    )
    dim_feat = feat_extractor.dim_feat

    # 2. Build the model

    # Load from YAML file

    model = build_model(
        params=params,
        batchsize=batchsize,
        dim_feat=dim_feat,
        time_steps=params.data['target_length_in_secs'] * params.data.signal.sampling_rate // params.train.feature.hop_size,
    )

    _, epoch_loaded_1 = load_model_checkpoint(
        model, params_train['epoch_loaded'], checkpoint_dir)

    # 3. Create the dataset
    tfrecord_list = {
        'train': Path(params.data['path_tfrecord']) / params.data['tfrecord_datalist_name']['train'],
        'val':  Path(params.data['path_tfrecord']) / params.data['tfrecord_datalist_name']['val'],
    }

    ds_train, batches_train = create_dataset(
        tfrecord_list['train'],
        batchsize=batchsize,
        is_shuffle=True,
        # num_samples = params.train['num_samples']['train'],
    )
    ds_val, batches_val = create_dataset(
        tfrecord_list['val'],
        batchsize=batchsize,
        is_shuffle=False,
        # num_samples = params.train['num_samples']['val'],
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
    if params.train.lr_schedule == 'cosine':
        lr_schedule = WarmUpCosineDecay(
            initial_lr = float(params_train['initial_lr']),
            total_steps = params_train['epochs'] * batches_train,
            warmup_steps =params_train['warmup_epochs'] * batches_train,
            alpha=1e-5,
            initial_step=epoch_loaded_1 * batches_train,)
    else:
        lr_schedule = float(params_train['initial_lr'])
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
            'total_batches': {'train': batches_train, 'val': batches_val},
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
        else:
            # print(f"Epoch {epoch} log: {log_epoch}")
            save_train_log(train_log, log_path)

        # 5. Save model weights
        os.makedirs("checkpoints", exist_ok=True)
        model.save_weights(f"{checkpoint_dir}/checkpoints/model_checkpoint_ep{epoch}")
