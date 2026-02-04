""" Evaluate for SE task"""
import os
import re
import logging
from pathlib import Path
import numpy as np
import soundfile as sf
import tensorflow as tf
from soundkit.defines import SKTaskParams
from soundkit.utils.np_feature_utils import FeatureExtractor_np
from soundkit.utils.pyaudio_animation import AudioShowClass
from soundkit.utils.calculate_feat_stats import load_feat_stats
from soundkit.utils.TFLiteAudioModel import TFLiteAudioModel
from soundkit.utils.generate_feature_c_files import generate_feature_c_files
from soundkit.utils.basic_dsp import DCRemover
from soundkit.utils.np_stft import StreamingISTFT
from soundkit.utils.converter_fix_point import (
        fakefix_tf,
        int2str_array
    )
from soundkit.utils.tf_stft import gen_stft_win
from soundkit.utils.feature_utils import FeatureExtractor
from soundkit.utils.mel import gen_mel_c
from soundkit.utils.erb import ERB
from soundkit.utils.plot_api import plot_spectrograms, fig_to_image
from soundkit.utils.audio import audio_read
from soundkit.utils.dnsmos_batch import DNSMOS_Batch
from .export import export


erb = ERB(
    erb_subband_1=65,
    erb_subband_2=64,
    platform="numpy")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

log = logging.getLogger(__name__)


def evaluate(params: SKTaskParams):
    """
    Deploy a TFLite model to neuralSPOT and install dependencies.

    Args:
        params (SKTaskParams): Task parameters
    """
    # === export TFLite File ===
    tflite_filename_src = f"{params.name}_{params.evaluate['dtype']}.tflite"

    tflite_path_src = Path(params.demo['tflite_dir']) / tflite_filename_src
    log.info(f"🧪 Exporting TFLite model from {tflite_path_src}")
    params.export['epoch_loaded'] = params.evaluate['epoch_loaded']
    params.export['tflite_dir'] = params.demo['tflite_dir']
    params.export["calibration_samples"] = params.evaluate["calibration_samples"]
    params.export["num_frames_infer"] = params.demo["num_frames_infer"]
    params.export["dtype"] = params.evaluate["dtype"]
    export(params)

    checkpoint_dir = f"{params.train['path']['checkpoint_dir']}"

    if params.train.feature.type=='hybrid':
        mel_bins = params.train.feature.n_mels
    else:
        mel_bins = params.train.feature.bins
    feat_extractor = FeatureExtractor_np(
        feat_type=params.train.feature.type,
        frame_len=params.train.feature.frame_size,
        hop_len=params.train.feature.hop_size,
        fft_len=params.train.feature.fft_size,
        sampling_rate=params.data.signal.sampling_rate,
        mel_bins=mel_bins,
        exp_complex=params.train.feature.exp_complex,
        scale=params.train.feature.scale,
        platform="numpy"
    )
    hop_size = params.train.feature.hop_size

    if params.train.standardization:
        stats = load_feat_stats(checkpoint_dir, 'stats.pkl')
    else:
        stats = None

    model_tflite = TFLiteAudioModel(
        interpreter_path=str(tflite_path_src),
        dtype=params.evaluate['dtype'],
    )

    class SEModel:
        """VAD model class in tflite for processing audio input and returning VAD output."""
        def __init__(
                self,
                model_tflite: TFLiteAudioModel,
                feat_extractor: FeatureExtractor_np,
                stats: dict | None = None,
                num_lookahead: int = 0,
                return_mask: bool = False):
            self.return_mask = return_mask
            self.num_lookahead = num_lookahead
            self.model_tflite = model_tflite
            self.stats = stats
            self.feat_extractor = feat_extractor
            self.specs = []
            self.istft = StreamingISTFT(
                frame_len=params.train.feature.frame_size,
                hop_len=params.train.feature.hop_size,
                fft_len=params.train.feature.fft_size,
            )

            if params.train.feature.type == 'erb_complex':
                self.erb = erb

            if num_lookahead > 0:
                _, z_spec = feat_extractor(
                    np.zeros(hop_size, dtype=np.float32))  # Warm up the feature extractor
                for i in range(num_lookahead):
                    self.specs.append(z_spec.copy())
            if params.data['signal']['dc_removal']:
                self.dc_remover = DCRemover()

        def __call__(self,
                     inputs: np.ndarray # input from microphone
                    ) -> np.ndarray: # output to AudioShowClass
            """Process input audio signal and return VAD output."""
            inputs=inputs.flatten()
            if params.data['signal']['dc_removal']:
                inputs = self.dc_remover.process(inputs)
            features,spec_update = self.feat_extractor(inputs)
            self.specs.append(spec_update)
            spec = self.specs.pop(0)
            # if self.stats is not None:
            #     features = (features - self.stats['nMean_feat']) * self.stats['nInvStd']
            if params.train['standardization']:
                # Standardize features
                if params.train['standardization_type'] in ["mve", "mean", "std"]:
                    mean_stats = self.stats['nMean_feat']
                    inv_std_stats = self.stats['nInvStd']
                    features = (features - mean_stats) * inv_std_stats
                elif params.train['standardization_type'] == "constant":
                    features = features / 32
            else:
                # No standardization, use raw features
                features = features
            # input to the tflite model
            features_t = features.reshape((1, 1, -1)) # reshape to (batch_size, time_steps, dim_feat)

            # reshape to (batch_size, time_steps, dim_feat, 2) for complex input
            if np.iscomplexobj(features_t):
                features_t = np.stack(
                    (features_t.real, features_t.imag), axis=-1)

            # features =fakefix_tf(features, 32, 21).numpy()

            tfmask = self.model_tflite(features_t)

            pcm_out, spec_en = self.post_procsessing(tfmask, spec)

            if self.return_mask:
                return pcm_out.reshape((-1,1)), tfmask, spec, spec_en, features
            else:
                return pcm_out.reshape((-1,1))

        def post_procsessing(
                self,
                tfmask: np.ndarray,
                spec: np.ndarray) -> np.ndarray:
            """
            post processing after getting mask from tflite model
            Args:
                tfmask (np.ndarray): output mask from tflite model
                spec (np.ndarray): input complex spectrogram
            Returns:
                np.ndarray: time-domain waveform after ISTFT
            """

            if params.train.feature.type == 'erb_complex':
                if 1:
                    tfmask = np.transpose(tfmask, axes=[0, 3, 1, 2]) # (B,T,F_erb,2) -> (B,2, T, F_erb)
                    tfmask = erb.bs(tfmask)
                    tfmask = np.transpose(tfmask, axes=[0, 2, 3, 1])  # (B,2, T, F_erb) -> (B,T,F_erb,2)

                    # set minimum value on tfmask to avoid too small values

                    amp = np.sqrt(tfmask[...,0]**2 + tfmask[...,1]**2)
                    
                    min_amp = 0.0
                    amp = np.where(amp < min_amp, min_amp, amp)
                    phase = np.angle(tfmask[:, :, :, 0] + 1j * tfmask[:, :, :, 1])
                    real = amp * np.cos(phase)
                    imag = amp * np.sin(phase)
                    tfmask = np.stack([real, imag], axis=-1)
            
                    tfmask = tfmask[:, :, :, 0] + 1j * tfmask[:, :, :, 1]
                else:
                    # 1. Decode to full spectrogram first (B, 2, T, F_full)
                    tfmask = np.transpose(tfmask, axes=[0, 3, 1, 2])
                    tfmask = erb.bs(tfmask)
                    tfmask = np.transpose(tfmask, axes=[0, 2, 3, 1])
                    
                    # 2. Convert to Complex
                    c_mask = tfmask[:, :, :, 0] + 1j * tfmask[:, :, :, 1]
                    
                    # 3. Smooth ONLY the Magnitude
                    # This prevents complex cancellation (which causes the 0 output)
                    from scipy.ndimage import uniform_filter1d
                    mag = np.abs(c_mask)
                    
                    # Smooth magnitude over time (axis 1) using a moving average
                    # A size of 3 or 5 frames is enough to kill musical noise without killing speech
                    mag_smoothed = uniform_filter1d(mag, size=3, axis=1)
                    
                    # 4. Re-apply smoothed magnitude to the original phase
                    # This ensures the complex mask NEVER cancels itself out
                    phase_mask = c_mask / (mag + 1e-12)
                    c_mask = mag_smoothed * phase_mask
                    
                    # 5. Apply a hard floor (e.g., -30dB)
                    # This is the final guard against "almost 0" values
                    floor = 0.03
                    tfmask = np.where(np.abs(c_mask) < floor, floor * phase_mask, c_mask)
                

            elif params.train.feature.type == 'erb_mag':
                tfmask = erb.bs(tfmask[..., 0])

            tfmask = tfmask.flatten()

            spec_en = spec * tfmask
            pcm_out = self.istft.process(spec_en)

            return pcm_out, spec_en

        def reset(self):
            """Reset the internal state of the model."""
            self.feat_extractor.reset()
            self.specs = []
            self.istft.reset()
            self.model_tflite.reset()
            if self.num_lookahead > 0:
                _, z_spec = feat_extractor(
                    np.zeros(hop_size, dtype=np.float32))  # Warm up the feature extractor
                for i in range(self.num_lookahead):
                    self.specs.append(z_spec.copy())
            if hasattr(self, 'dc_remover'):
                self.dc_remover.reset()

    # 4. Run the AudioShowClass
    se_model = SEModel(
        model_tflite=model_tflite,
        feat_extractor=feat_extractor,
        stats=stats,
        num_lookahead=params.train.num_lookahead,
        return_mask=True,
    )
    os.makedirs(params.evaluate.data.result_folder, exist_ok=True)
    for file in params.evaluate.data.files:
        wavfile = os.path.join(params.evaluate.data.dir, file)
        y = audio_read(
            wavfile,
            sample_rate=params.data.signal.sampling_rate)

        se_model.reset()
        hopsize= params.train.feature.hop_size

        outs_pcm = []
        outs_tfmsk = []
        out_specs = []
        out_features = []
        out_specs_en = []
        for i in range(0, len(y), hopsize):
        #     print( f"Processing {i}/{len(y)} samples", end='\r')
            chunk = y[i:i+hopsize]
            if len(chunk) < hopsize:
                chunk = np.pad(
                    chunk,
                    (0, hopsize - len(chunk)),
                    mode='constant')
            pcm_out, tfmask, spec, spec_en, features = se_model(chunk)
            outs_pcm.append(pcm_out)
            outs_tfmsk.append(tfmask)
            out_features.append(features)
            out_specs.append(spec)
            out_specs_en.append(spec_en)
        outs_pcm = np.concatenate(outs_pcm, axis=0).reshape((-1,))

        outs_tfmsk = np.concatenate(outs_tfmsk, axis=0)
        out_specs = np.stack(out_specs, axis=0)
        out_specs_en = np.stack(out_specs_en, axis=0)
        out_features = np.stack(out_features, axis=0)
        if outs_tfmsk.ndim == 4:
            outs_tfmsk = outs_tfmsk[:,0]
            real_mask = outs_tfmsk[...,0]
            imag_mask = outs_tfmsk[...,1]
            features = 20 * np.log10(np.abs(out_features))
        elif outs_tfmsk.ndim ==3:
            real_mask = outs_tfmsk
            imag_mask = None
        
        logpspec_sn = 20.0 * np.log10(
            np.maximum(
                np.abs(out_specs),
                1e-8))
        logpspec_en = 20.0 * np.log10(
            np.maximum(
                np.abs(out_specs_en),
                1e-8))

        filename = re.sub(r'(\.wav$|\.flac$)', '', file)
        filename = filename + f"_dtype{params.evaluate['dtype']}"
        save_path = os.path.join(params.evaluate.data.result_folder, f"spectrograms_{filename}.pdf")

        plot_spectrograms(
            [logpspec_sn.T, logpspec_en.T, features.T, real_mask.T, imag_mask.T if imag_mask is not None else None],
            titles=['Noisy Spectrogram', 'Enhanced Spectrogram', 'Features', 'Real Mask', 'Imag Mask' if imag_mask is not None else None],
            vmin_vmax=[(-80, 10), (-80, 10), (np.min(features), np.max(features)), (np.min(real_mask), np.max(real_mask)), (np.min(imag_mask), np.max(imag_mask)) if imag_mask is not None else None],
            save_path=save_path,
            show_fig=False
        )

        sf.write(
            os.path.join(params.evaluate.data.result_folder, f"{filename}_en.wav"),
            outs_pcm,
            samplerate=params.data.signal.sampling_rate)
        sf.write(
            os.path.join(params.evaluate.data.result_folder, f"{filename}_sn.wav"),
            y,
            samplerate=params.data.signal.sampling_rate)
        print("Enhanced file saved to {}".format(
            os.path.join(params.evaluate.data.result_folder, f"{filename}_en.wav")))
    runner = DNSMOS_Batch(use_gpu=False, batch_size=4)
    result_folder = params.evaluate.data.result_folder
    runner.run_folder(result_folder)