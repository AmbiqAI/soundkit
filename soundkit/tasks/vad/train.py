import os
import datetime
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import tensorflow as tf
from .datasets import create_dataset
from ...defines import SKTaskParams
from ...utils.download_tf_model import save_train_log, load_train_log
from ...utils.download_tf_model import build_model, load_model_checkpoint
from ...utils.feature_utils import FeatureExtractor
from ...utils.losses import LossFactory
from ...utils.calculate_feat_stats import feat_stats_estimator
from ...utils.lookaheadBuffer import LookaheadBuffer
from ...utils.WarmUpCosineDecay import WarmUpCosineDecay
from ...utils.tf_complex_utils import complex_to_realarray
from ...utils.plot_api import plot_spectrograms
from ...utils.tf_basic_math import tf_log10_eps
from ...utils.ConfusionMatrixMetric import ConfusionMatrixMetric
from ...utils.calculate_feat_stats import mean_varinace_norm

@tf.function
def train_step(
        net: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        loss_fn: Any,
        batch: dict[str, tf.Tensor],
        training: bool = True,):
    """Perform a single training step."""

    feat_sn = batch["feat_sn"]
    vad = batch['vad']

    with tf.GradientTape() as tape:
        est = net(feat_sn, training=training)
        est = tf.math.softmax(est, axis=-1)
        loss = loss_fn(vad, est)

    if training:
        gradients = tape.gradient(loss, net.trainable_variables)
        gradients_clips = [ grad
                            for grad in gradients ]

        optimizer.apply_gradients(
                    zip(gradients_clips,
                        net.trainable_variables))
    return loss, est

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

    total_batches = config["total_batches"]['train'] if training else config["total_batches"]['val']

    total_steps=total_batches * epoch

    train_summary_writer = config["train_summary_writer"]
    train_tag = "train" if training else "val"

    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    loss_metric = tf.keras.metrics.Mean()
    confused_metric = ConfusionMatrixMetric(num_classes=2)
    # Initialize left-over state buffers for streaming STFT
    states_audio_sn = tf.random.uniform(
        [batchsize, stft_feat["frame_size"] - stft_feat["hop_size"]],
        minval=-1.0,
        maxval=1.0,
        dtype=tf.float32
    )
    model.reset_states()
    for step, batch in enumerate(dataset):

        audio_sn, _, vad = batch

        if model.stride_time > 1:
            vad = vad[:,::model.stride_time]
        # Extract features using streaming state
        feat_sn, spec_sn, states_audio_sn = feat_extractor(
            audio_sn, states=states_audio_sn)
        if params.train.reset_every_batch:
            # Reset the state buffer for the next batch
            states_audio_sn = tf.random.uniform(
                [batchsize, stft_feat["frame_size"] - stft_feat["hop_size"]],
                minval=-1.0,
                maxval=1.0,
                dtype=tf.float32
            )
            model.reset_states()
        if params.train['standardization']:
            # Standardize features
            feat_sn_norm = mean_varinace_norm(feat_sn, stats['nMean_feat'], stats['nInvStd'])
        else:
            # No standardization, use raw features
            feat_sn_norm = feat_sn

        if feat_extractor.feat_type=="spec":
            feat_sn_norm = complex_to_realarray(feat_sn_norm)
            batch_data = {
                "feat_sn": feat_sn_norm,
                "vad": vad
            }
        else:
            batch_data = {
                "feat_sn": feat_sn_norm,
                "vad": vad
            }

        # Perform one training or val step
        loss, pred = train_step(
            model,
            optimizer,
            loss_fn,
            batch_data,
            training=training)
        loss_metric.update_state(loss)
        acc_metric.update_state(vad, pred)
        confused_metric.update_state(vad, pred)

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
            f"    Acc: {acc_metric.result()} | ",
            end="\r",
            flush=True
        )

        if step == total_batches - 1:
            conf_matrix = confused_metric.result().numpy()
            print(
                f"\n[{train_tag}] Step {step + 1}/{total_batches}\n"
                f"  Loss        : {loss_metric.result():.4f}\n"
                f"  Accuracy    : {acc_metric.result():.4f}\n"
                f"  Confusion Matrix (row-normalized):\n{conf_matrix}\n",
                flush=True)

        if not training:
            if step == 0:
                idx = 10
                mask = pred[idx,:,1]
                pspec_sn = 20*tf_log10_eps( tf.abs(spec_sn[idx])).numpy()

                if params.train['feature']['type'] in ('mel', 'logpspec', 'hybrid'):
                    feat_sn = 10* feat_sn[idx].numpy()
                elif params.train['feature']['type'] in ('pspec', 'spec'):
                    feat_sn = 20*tf_log10_eps( tf.abs(feat_sn[idx])).numpy()
                feat_sn = feat_sn[::model.stride_time]

                fig = plot_spectrograms(
                    images=[pspec_sn.T, feat_sn.T],
                    titles=[ "noisy logspec", "feat"],
                    vmin_vmax=[(-80, 10), (-80, 10)],
                    show_colorbar=True,
                    show_fig=False       # set False if only saving
                )

                plt.plot(mask * 200)
                plt.plot(vad[idx] * 200)
                # plt.show()
                from ...utils.plot_api import fig_to_image
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

    return loss_metric, acc_metric, confused_metric

def train(params: SKTaskParams):
    """Train beat task model with given parameters.

    Args:
        params (HKTaskParams): Task parameters
    """
    print(f"Training VAD model with params: {params} and more")

    params_train = params.train
    checkpoint_dir = f"{params.train['path']['checkpoint_dir']}"
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tfboard_dir = f"{params.train['path']['tensorboard_dir']}/logs/{current_time}"
    train_summary_writer = tf.summary.create_file_writer(tfboard_dir)
    batchsize = params.train['batchsize']

    # 1. Define feature extractor
    feat_extractor = FeatureExtractor(
        params=params,
    )
    dim_feat = feat_extractor.dim_feat

    # 2. Build the model

    # Load from YAML file
    is_complex = True if params_train['feature']['type'] =='spec' else False

    model = build_model(
        params,
        batchsize,
        dim_feat,
        time_steps = params.data['target_length_in_secs'] * 100,
        complex_input=is_complex)

    _, epoch_loaded_1 = load_model_checkpoint(
        model, params_train['epoch_loaded'], checkpoint_dir)

    # import pickle

    # with open('array_list.pkl', 'rb') as f:
    #         reloaded_list = pickle.load(f)

    # for u, v in zip(model.trainable_variables, reloaded_list):
    #     u.assign(v)
    #     print(u.shape, v.shape)

    # 3. Create the dataset
    tfrecord_list = {
        'train': Path(params.data['path_tfrecord']) / params.data['tfrecord_datalist_name']['train'],
        'val':  Path(params.data['path_tfrecord']) / params.data['tfrecord_datalist_name']['val'],
    }

    ds_train, batches_train = create_dataset(
        tfrecord_list['train'],
        batchsize=batchsize,
        hop_size = params.train.feature.hop_size,
        is_shuffle=True,
    )
    ds_val, batches_val = create_dataset(
        tfrecord_list['val'],
        batchsize=batchsize,
        hop_size = params.train.feature.hop_size,
        is_shuffle=False,
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
        **params.train["loss_function"]["params"])

    lr_schedule = WarmUpCosineDecay(
        initial_lr = float(params_train['initial_lr']),
        total_steps = params_train['epochs'] * batches_train,
        warmup_steps =params_train['warmup_epochs'] * batches_train,
        alpha=1e-5,
        initial_step=epoch_loaded_1 * batches_train,)

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
            loss_metric, acc_metric, confused_metric = run_epoch(
                train_config,
                ds_train,
                training=True,
                epoch=epoch,
            )
            log_epoch["train_loss"] = float(loss_metric.result().numpy())
            log_epoch["train_acc"] = float(acc_metric.result().numpy())
            if train_summary_writer is not None:
                with train_summary_writer.as_default():
                    tf.summary.scalar(f"train/loss", loss_metric.result(), step=epoch)
                    tf.summary.scalar(f"train/acc", acc_metric.result(), step=epoch)

        # Validation phase
        loss_metric, acc_metric, confused_metric = run_epoch(
            train_config,
            ds_val,
            training=False,
            epoch=epoch,
        )
        log_epoch["val_loss"] = float(loss_metric.result().numpy())
        log_epoch["val_acc"] = float(acc_metric.result().numpy())
        if train_summary_writer is not None:
            with train_summary_writer.as_default():
                tf.summary.scalar(f"val/loss", loss_metric.result(), step=epoch)
                tf.summary.scalar(f"val/acc", acc_metric.result(), step=epoch)

        train_log[epoch] = log_epoch

        save_train_log(train_log, log_path)

        # 5. Save model weights
        os.makedirs("checkpoints", exist_ok=True)
        model.save_weights(f"{checkpoint_dir}/checkpoints/model_checkpoint_ep{epoch}")
