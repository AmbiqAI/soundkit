import tensorflow as tf
from typing import Union
from .tf_stft import tf_stft
from .tf_basic_math import tf_log10_eps
from .mel import gen_mel_bank
from .converter_fix_point import fakefix_tf
from .mel_spec_gen import melspec_gen
from ..defines import SKTaskParams
class FeatureExtractor:
    """
    Feature extractor for audio signals using a dispatch map.
    """

    def __init__(
            self,
            params: SKTaskParams):

        self.signal_config = params.data['signal']
        feat_type = params.train['feature']["type"]
        num_bins =  params.train['feature']["bins"]
        self._extractors = {
            "spec": self._extract_spec,
            "pspec": self._extract_pspec,
            "logpspec": self._extract_logpspec,
            "mel": self._extract_mel,
            "logpspec_mel": self._extract_logpspec_mel,
        }

        if feat_type not in self._extractors:
            raise ValueError(f"Unsupported feature type: {feat_type}")

        self._extract_fn = self._extractors[feat_type]
        self.stft_exec = lambda x, states: tf_stft(
            x,
            frame_length=self.signal_config["frame_size"],
            frame_step=self.signal_config["hop_size"],
            fft_length=self.signal_config["fft_size"],
            states=states,
        )

        if feat_type == "mel":
            fbanks = gen_mel_bank(
                    fftsize         = self.signal_config["fft_size"],
                    nfilt           = num_bins,
                    sample_rate     = self.signal_config["sampling_rate"],)
            self.mel_filter = tf.constant(fbanks.T, dtype=tf.float32)
        elif feat_type == "logpspec_mel":
            fbanks = melspec_gen(
                samplingRate=self.signal_config["sampling_rate"],
                n_fft=self.signal_config["fft_size"],
                n_mels=32,
                thresh_mel=50)
            self.mel_filter = tf.constant(fbanks.T, dtype=tf.float32)

    def __call__(
            self,
            audio_sn: tf.Tensor,
            states : Union[tf.Tensor, None] = None) -> tf.Tensor:
        """Extract features from audio using configured extractor."""
        overlap=self.signal_config["frame_size"] - self.signal_config["hop_size"]
        states_udpate = tf.identity(audio_sn[:, -overlap:])

        feat, spec = self._extract_fn(audio_sn, states)

        return feat, spec, states_udpate

    def _extract_spec(
            self,
            audio_sn: tf.Tensor,
            states: Union[tf.Tensor, None]) -> tf.Tensor:

        spec =  self.stft_exec(
            audio_sn,
            states=states,
        )

        return spec, spec

    def _extract_logpspec(
            self,
            audio_sn: tf.Tensor,
            states: Union[tf.Tensor, None]) -> tf.Tensor:

        feat, spec = self._extract_spec(
            audio_sn,
            states=states,
        )

        return tf_log10_eps(tf.abs(spec)), spec

    def _extract_pspec(
            self,
            audio_sn: tf.Tensor,
            states: Union[tf.Tensor, None]) -> tf.Tensor:

        feat, spec = self._extract_spec(
            audio_sn,
            states=states,
        )

        return tf.abs(feat), spec

    def _extract_mel(
            self,
            audio_sn: tf.Tensor,
            states: Union[tf.Tensor, None]) -> tf.Tensor:

        pspec, spec = self._extract_pspec(
            audio_sn,
            states=states,
        )

        mel_spec = tf.matmul(
            pspec**2,
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
            pspec**2,
            self.mel_filter,
        )
        mel_spec = tf_log10_eps(mel_spec)

        return mel_spec, spec