from pathlib import Path
import tensorflow as tf
import torch
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
import numpy as np

from ...utils.audio import audio_read
from .datasets import create_dataset
from ...defines import SKTaskParams
from ...utils.download_tf_model import build_model, load_model_checkpoint
from ...utils.feature_utils import FeatureExtractor
from ...utils.calculate_feat_stats import feat_stats_estimator
from ...utils.lookaheadBuffer import LookaheadBuffer
from ...utils.tf_stft import tf_istft, StreamingSTFT, StreamingISTFT
from ...utils.tf_complex_utils import polar_to_complex
from ...utils.tflite_convert import tflite_convert, warp_tf_model
from ...utils.tf_copy_model import copy_model_weights
from time import time
def evaluate(params: SKTaskParams):
    """Evaluate SE task model with given parameters.

    Args:
        params (HKTaskParams): Task parameters
    """
    print(f"Evaluating SE model with params: {params} and more")

    params_evaluate = params.evaluate
    num_lookahead = params.train['num_lookahead']
    signal = params.data['signal']

    model_dir = Path(params.train['path']['models_trained']) / params.name

    batchsize_train = params.train['batchsize']
    batchsize = 1
    dim_feat = params.train['feature']['bins']

    # 1.1. Build the model

    # Load from YAML file
    model_train = build_model(
        params, batchsize=batchsize_train, dim_feat=dim_feat)
    load_model_checkpoint(
        model_train, params_evaluate['epoch_loaded'], model_dir)

    model = build_model(
        params, batchsize=1, dim_feat=dim_feat, export=True)
    copy_model_weights(
        model_dst=model,
        model_src=model_train)

    # 2. Create the dataset
    tfrecord_list = {
        'test':  Path(params.data['path_tfrecord']) / params.data['tfrecord_datalist_name']['test'],
    }

    dataset, batches = create_dataset(
        tfrecord_list['test'],
        batchsize=batchsize,
        is_shuffle=True,
    )

    # 3. Define feature extractor
    feat_extractor = FeatureExtractor(
        params=params,
    )

    # 4. Compute feature statistics for standardization
    stats = feat_stats_estimator(
        dataset,
        batches,
        dim_feat,
        folder_nn=model_dir,
        feat_extractor=feat_extractor,)

    # Initialize left-over state buffers for streaming STFT
    states_audio_sn = tf.zeros(
        [batchsize, signal["frame_size"] - signal["hop_size"]],
        dtype=tf.float32
    )

    num_fft_bins = signal["fft_size"] // 2 + 1
    buffer_sn = LookaheadBuffer(
        num_lookahead=num_lookahead,
        feature_dim=num_fft_bins,
        batchsize=batchsize)

    nn_train = warp_tf_model(
        model,
        time_steps=1,
        dim_feat=dim_feat,)

    dtype="float32"
    tflite_fp16_model = tflite_convert(
        nn_train,
        dtype=dtype,
        path_tflite=f'./tflite/nnse_{dtype}.tflite')
    interpreter = tf.lite.Interpreter(
        model_content=tflite_fp16_model,
        # model_path=f'./tflite/nnse_{dtype}.tflite'
        )
    interpreter.allocate_tensors()  # Needed before execution!

    # Get input and output tensors.
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_shape = input_details['shape']
    
    audio_sn = audio_read("./wavs/se/test_wavs/keyboard_steak.wav")
    
    audio_sn = tf.constant(audio_sn, dtype=tf.float32)
    audio_sn = tf.reshape(audio_sn, (1, -1))
    
    feat_sn, spec_sn, states_audio_sn = feat_extractor(
            audio_sn, states=states_audio_sn)

    
    if params.train['standardization']:
        # Standardize features

        mean_stats = stats['nMean_feat']
        inv_std_stats = stats['nInvStd']
        feat_sn_norm = (feat_sn - mean_stats) * inv_std_stats

    if dtype in ("int8", "int16"):
        input_scale, input_zero_point = input_details["quantization"]
        nfeat_np = feat_sn_norm.numpy() / input_scale + input_zero_point
        if dtype == "int8":
            nfeat_np = nfeat_np.astype(np.int8)
        elif dtype == "int16":
            nfeat_np = nfeat_np.astype(np.int16)
    else:
        nfeat_np = feat_sn_norm.numpy().astype(np.float32)

    STFT = StreamingSTFT()
    ISTFT = StreamingISTFT()
    ss = STFT.process_frame(
        tf.ones(shape=(signal['hop_size'],), dtype=tf.float32))
    ss = tf.reshape(ss, (1, -1))

    out = []
    time_start = time()
    for i in range(nfeat_np.shape[1]):
        print(f"\rProcessing frame {i}/{nfeat_np.shape[1]}", end = '')

        
        
        input_data = nfeat_np[:,i:i+1,:]
        interpreter.set_tensor(
            input_details['index'],
            input_data)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details['index'])
        out += [output_data]
    print(f"\nProcessing time: {(time() - time_start)/nfeat_np.shape[1]} seconds")
    out = np.concatenate(out, axis=1)
    if dtype in ("int8", "int16"):
        output_scale, output_zero_point = output_details["quantization"]
        out = (out - output_zero_point).astype(np.float32) * output_scale

    tfmask = tf.constant(
        out,
        dtype=tf.float32)

    spec_sn_delay = buffer_sn.apply(spec_sn)
    pspec_sn_delay = tf.abs(spec_sn_delay)
    phase_sn_delay = tf.math.angle(spec_sn_delay)

    pspec_en_delay = tfmask * pspec_sn_delay
    spec_en_delay = polar_to_complex(
            pspec_en_delay, phase_sn_delay)
        
    audio_en = tf_istft(
        spec_en_delay,
        signal['frame_size'],
        signal['hop_size'],
        signal['fft_size'])
    