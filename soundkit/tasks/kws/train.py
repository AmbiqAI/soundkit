"""Train keyword spotting model with given parameters."""
import os
import logging
import datetime
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import tensorflow as tf
from soundkit.defines import SKTaskParams
from soundkit.utils.download_tf_model import save_train_log, load_train_log
from soundkit.utils.download_tf_model import build_model, load_model_checkpoint
from soundkit.utils.feature_utils import FeatureExtractor
from soundkit.utils.losses import LossFactory
from soundkit.utils.calculate_feat_stats import feat_stats_estimator
from soundkit.utils.WarmUpCosineDecay import WarmUpCosineDecay
from soundkit.utils.tf_complex_utils import complex_to_realarray
from soundkit.utils.plot_api import plot_spectrograms
from soundkit.utils.tf_basic_math import tf_log10_eps
from soundkit.utils.ConfusionMatrixMetric import ConfusionMatrixMetric
from soundkit.utils.calculate_feat_stats import mean_varinace_norm
from soundkit.utils.plot_api import fig_to_image
from .datasets import create_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)

@tf.function
def train_step(
        net: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        loss_fn: Any,
        batch: dict[str, tf.Tensor],
        training: bool = True,):
    """Perform a single training step."""

    feat_sn = batch["feat_sn"]
    kws = batch['kws']

    with tf.GradientTape() as tape:
        est = net(feat_sn, training=training)
        est = tf.math.softmax(est, axis=-1)
        loss = loss_fn(kws, est)

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

    def reset_nn_states(model: tf.keras.Model):
        """Reset the model states."""
        states_audio_sn = tf.random.uniform(
            [batchsize, stft_feat["frame_size"] - stft_feat["hop_size"]],
            minval=-1.0,
            maxval=1.0,
            dtype=tf.float32
        )* 10**-5
        model.reset_states()
        return states_audio_sn
    states_audio_sn = reset_nn_states(model)
    for step, batch in enumerate(dataset):

        audio_sn, _, kws = batch

        if model.stride_time > 1:
            kws = kws[:,::model.stride_time]
        # Extract features using streaming state
        feat_sn, spec_sn, states_audio_sn = feat_extractor(
            audio_sn, states=states_audio_sn)
        if params.train.reset_states_every_batch:
            # Reset the state buffer for the next batch
            states_audio_sn = reset_nn_states(model)

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
                "kws": kws
            }
        else:
            batch_data = {
                "feat_sn": feat_sn_norm,
                "kws": kws
            }

        # Perform one training or val step
        loss, pred = train_step(
            model,
            optimizer,
            loss_fn,
            batch_data,
            training=training)
        loss_metric.update_state(loss)
        acc_metric.update_state(kws, pred)
        confused_metric.update_state(kws, pred)

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


        if step % 100 == 0:
            idx = 0
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
            plt.plot(kws[idx] * 200)
            # plt.show()

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
        params (SKTaskParams): Task parameters
    """
    log.info(f"Training KWS model with params: {params} and more")

    params_train = params.train
    checkpoint_dir = f"{params.train['path']['checkpoint_dir']}"
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tfboard_dir = f"{params.train['path']['tensorboard_dir']}/logs/{current_time}"
    train_summary_writer = tf.summary.create_file_writer(tfboard_dir)
    batchsize = params.train['batchsize']
    if batchsize < 8:
        log.warning(
            "⚠️  Warning: Batch size is very small. This may lead to slow training and unstable gradients.\n\n"
            "💡 Consider increasing the batch size in your config.yaml file for better performance:\n\n"
            "    train:\n"
            "      batchsize: 32\n\n"
            "   or \n\n"
            "     soundkit -t se -m train -c configs/se/se.yaml train.batchsize=32\n\n"
            "Or set it to a value that fits your GPU memory.\n"
            "Or choose the highest value that fits within your available memory.\n"
        )
    # 1. Define feature extractor
    feat_extractor = FeatureExtractor(
        params=params,
    )
    dim_feat = feat_extractor.dim_feat

    # 2. Build the model

    # Load from YAML file

    time_steps = int(params.data['target_length_in_secs'] * params.data.signal.sampling_rate //  params.train.feature.hop_size)
    model = build_model(
        params,
        batchsize,
        dim_feat,
        time_steps=time_steps)

    _, epoch_loaded_1 = load_model_checkpoint(
        model, params_train['epoch_loaded'], checkpoint_dir)


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
        params=params.train["loss_function"]["params"])

    # 6. Define learning rate schedule
    if params.train.lr_schedule=='cosine':
        lr_schedule = WarmUpCosineDecay(
            initial_lr = float(params_train['initial_lr']),
            total_steps = params_train['epochs'] * batches_train,
            warmup_steps =params_train['warmup_epochs'] * batches_train,
            alpha=1e-5,
            initial_step=epoch_loaded_1 * batches_train,)
    elif params.train.lr_schedule=='constant':
        lr_schedule=float(params_train['initial_lr'])
    else:
        raise ValueError(
            f"Learning rate schedule {params.train.lr_schedule} is not supported. Use 'cosine' or 'constant' instead."
        )
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
        log.info(f"Epoch {epoch}/{params_train['epochs']}\n")

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
