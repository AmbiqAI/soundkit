''' Feature extraction utilities for audio signals.
This module provides a `FeatureExtractor` class that can extract various types of audio features
from audio signals, such as spectrograms, mel spectrograms, and time-domain features.
'''
from typing import Union
import numpy as np
import tensorflow as tf
from soundkit.utils.tf_stft import tf_stft
from soundkit.utils.tf_stft import gen_stft_win
from soundkit.utils.tf_basic_math import tf_log10_eps
from soundkit.utils.mel import gen_mel_bank
from soundkit.utils.mel_spec_gen import melspec_gen
from soundkit.defines import SKTaskParams

class FeatureExtractor:
    """
    Feature extractor for audio signals using a dispatch map.
    """

    def __init__(
            self,
            params: SKTaskParams):

        self.params = params
        feat_params = self.params.train['feature']
        self._extractors = {
            "spec": self._extract_spec, # spectrogram
            "pspec": self._extract_pspec, # power spectrogram
            "logpspec": self._extract_logpspec, # log power spectrogram
            "logampspec": self._extract_logampspec, # log amplitude spectrogram
            "mel": self._extract_mel, # log mel spectrogram
            "hybrid": self._extract_logpspec_mel, # hybrid spectrogram
            "time": self._extract_time, # time-domain features
        }

        if feat_params['type'] not in self._extractors:
            raise ValueError(f"Unsupported feature type: {feat_params['type']}")
        else:
            self.feat_type = feat_params['type']
        self._extract_fn = self._extractors[feat_params['type']]

        if feat_params["frame_size"] % feat_params["hop_size"] != 0:
            raise ValueError(
                 f"Frame size must be divisible by hop size. "
                f"Got frame_size={feat_params['frame_size']} and hop_size={feat_params['hop_size']}"
            )

        self.stft_exec = lambda x, states: tf_stft(
            x,
            frame_length=feat_params["frame_size"],
            frame_step=feat_params["hop_size"],
            fft_length=feat_params["fft_size"],
            states=states,
        )

        if feat_params['type'] == "mel":
            fbanks = gen_mel_bank(
                    fftsize         = feat_params["fft_size"],
                    nfilt           = feat_params['bins'],
                    sample_rate     = params.data['signal']["sampling_rate"],)
            self.mel_filter = tf.constant(fbanks.T, dtype=tf.float32)
            self.dim_feat = feat_params['bins']

        elif feat_params['type'] == "hybrid":
            if feat_params['bins_fft'] is None:
                raise ValueError("bins_fft must be specified for hybrid feature extraction")
            if feat_params['n_mels'] is None:
                raise ValueError("n_mels must be specified for hybrid feature extraction")

            fbanks = melspec_gen(
                samplingRate=params.data['signal']["sampling_rate"],
                n_fft=feat_params['fft_size'],
                n_mels=feat_params['n_mels'],
                thresh_mel=feat_params['bins_fft'])
            self.mel_filter = tf.constant(fbanks.T, dtype=tf.float32)
            self.mel_filter_inv = tf.linalg.pinv(self.mel_filter)

            self.dim_feat = fbanks.shape[0]

        elif feat_params['type'] == "time":
            self.window=gen_stft_win(
                win_size=feat_params['frame_size'],
                hop=feat_params['hop_size'])
            self.dim_feat = feat_params['frame_size']

        else:
            dim_feat = (feat_params['fft_size'] // 2) + 1
            self.mel_filter = tf.eye(
                dim_feat)
            self.dim_feat = dim_feat

    def __call__(
            self,
            audio_sn: tf.Tensor,
            states : Union[tf.Tensor, None] = None) -> tf.Tensor:
        """Extract features from audio using configured extractor."""

        feat_params = self.params.train['feature']
        overlap=feat_params["frame_size"] - feat_params["hop_size"]
        states_udpate = tf.identity(audio_sn[:, -overlap:])

        feat, spec = self._extract_fn(audio_sn, states)

        return feat, spec, states_udpate

    @property
    def dim(self) -> int:
        """Return the number of feature dimensions."""
        return self.dim_feat

    def _extract_spec(
            self,
            audio_sn: tf.Tensor,
            states: Union[tf.Tensor, None]) -> tf.Tensor:

        spec =  self.stft_exec(
            audio_sn,
            states=states,
        )

        return spec, tf.identity(spec)

    def _extract_pspec(
            self,
            audio_sn: tf.Tensor,
            states: Union[tf.Tensor, None]) -> tf.Tensor:

        feat, spec = self._extract_spec(
            audio_sn,
            states=states,
        )

        return tf.abs(feat)**2, spec

    def _extract_logampspec(
            self,
            audio_sn: tf.Tensor,
            states: Union[tf.Tensor, None]) -> tf.Tensor:

        feat, spec = self._extract_spec(
            audio_sn,
            states=states,
        )

        return tf_log10_eps(tf.abs(feat)), spec

    def _extract_logpspec(
            self,
            audio_sn: tf.Tensor,
            states: Union[tf.Tensor, None]) -> tf.Tensor:

        pspec, spec = self._extract_pspec(
            audio_sn,
            states=states,
        )

        return tf_log10_eps(pspec), spec

    def _extract_mel(
            self,
            audio_sn: tf.Tensor,
            states: Union[tf.Tensor, None]) -> tf.Tensor:

        pspec, spec = self._extract_pspec(
            audio_sn,
            states=states,
        )

        mel_spec = tf.matmul(
            pspec,
            self.mel_filter,
        )
        mel_spec = tf_log10_eps(mel_spec)

        return mel_spec, spec

    def _extract_logpspec_mel(
            self,
            audio_sn: tf.Tensor,
            states: Union[tf.Tensor, None]) -> tf.Tensor:

        pspec, spec = self._extract_pspec(
            audio_sn,
            states=states,
        )

        mel_spec = tf.matmul(
            pspec,
            self.mel_filter,
        )
        mel_spec = tf_log10_eps(mel_spec)

        return mel_spec, spec

    def _extract_time(
            self,
            audio_sn: tf.Tensor,
            states: Union[tf.Tensor, None]) -> tf.Tensor:

        feat_params = self.params.train['feature']

        if "fft_size" in feat_params:
            fft_size = feat_params["fft_size"]
        else:
            exp = int(np.ceil(np.log2(feat_params["frame_size"])))
            fft_size = 2 ** exp

        if states is None:
            states = tf.zeros(
                (audio_sn.shape[0],
                 feat_params["frame_size"]-feat_params["hop_size"]),
                dtype=tf.float32)

        audio_sn = tf.concat([states, audio_sn], axis=-1)

        frames = tf.signal.frame(
            audio_sn,
            frame_length = feat_params["frame_size"],
            frame_step = feat_params["hop_size"],
            pad_end = False,
            axis = -1)

        feat = frames * self.window
        spec = tf.signal.rfft(feat, [fft_size])

        return feat, spec
