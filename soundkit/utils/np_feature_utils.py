import numpy as np
from .np_stft import StreamingSTFT, StreamingISTFT

from soundkit.utils.mel import gen_mel_bank
from soundkit.utils.mel_spec_gen import melspec_gen

def log10_eps(val, eps = 2.0**-15):
    """
    log10 with minimum eps
    """
    return  np.log10(val+eps)

class FeatureExtractor_np:
    """
    Feature extractor for audio signals using numpy.
    Supports STFT, power spectrum, log power spectrum, and mel spectrogram.
    """
    def __init__(
            self,
            feat_type='logpspec',
            frame_len=480,
            hop_len=160,
            fft_len=512,
            sampling_rate=16000,
            mel_bins=72,
            stream=True):


        skip_fft = True if feat_type in ("time") else False

        self.stft_exec = StreamingSTFT(
            frame_len,
            hop_len,
            fft_len,
            stream=stream,
            skip_fft=skip_fft,
            )

        self.sampling_rate = sampling_rate
        self.feat_type=feat_type
        self._extractors = {
            "spec": self._extract_spec,
            "pspec": self._extract_pspec,
            "logpspec": self._extract_logpspec,
            "logampspec": self._extract_logampspec,
            "mel": self._extract_mel,
            "hybrid": self._extract_logpspec_mel,
            "time": self._extract_frames,
        }

        if feat_type == "mel":
            fbanks = gen_mel_bank(
                    fftsize         = fft_len,
                    nfilt           = mel_bins,
                    sample_rate     = sampling_rate,)
            self.mel_filter = fbanks.T
            self.dim_feat = mel_bins

        elif feat_type == "hybrid":

            fbanks = melspec_gen(
                samplingRate=sampling_rate,
                n_fft=fft_len,
                n_mels=mel_bins,
                thresh_mel=50)
            self.mel_filter = fbanks.T
            self.dim_feat = fbanks.shape[0]

        elif feat_type in ("spec", "pspec", "logpspec", "logampspec"):

            dim_feat = (fft_len // 2) + 1

            self.mel_filter = np.eye(
                dim_feat)
            self.dim_feat = dim_feat

        elif feat_type == "time":

            self.mel_filter = None
            self.dim_feat = frame_len

        else:
            raise ValueError(f"Unknown feature type: {feat_type}")

    @property
    def dim(self) -> int:
        """Return the number of feature dimensions."""
        return self.dim_feat
    def reset(self):
        """Reset the internal state of the feature extractor."""
        self.stft_exec.reset()
    def __call__(
            self,
            audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        extractor= self._extractors[self.feat_type]

        return extractor(audio)
    
    def _extract_frames(
            self,
            audio_sn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        spec = self.stft_exec.process(audio_sn)

        return spec, spec.copy()

    def _extract_spec(
            self,
            audio_sn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        spec = self.stft_exec.process(audio_sn)

        return spec, spec.copy()

    def _extract_pspec(
            self,
            audio_sn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        feat, spec = self._extract_spec(
            audio_sn
        )

        return np.abs(feat)**2, spec

    def _extract_logampspec(
            self,
            audio_sn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        spec, _ = self._extract_spec(
            audio_sn
        )

        return log10_eps(np.abs(spec)), spec

    def _extract_logpspec(
            self,
            audio_sn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        pspec, spec = self._extract_pspec(
            audio_sn
        )

        return log10_eps(pspec), spec

    def _extract_mel(
            self,
            audio_sn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        pspec, spec = self._extract_pspec(
            audio_sn,
        )

        mel_spec = np.matmul(
            pspec,
            self.mel_filter,
        )
        mel_spec = log10_eps(mel_spec)

        return mel_spec, spec

    def _extract_logpspec_mel(
            self,
            audio_sn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        pspec, spec = self._extract_pspec(
            audio_sn
        )

        mel_spec = np.matmul(
            pspec,
            self.mel_filter,
        )
        mel_spec = log10_eps(mel_spec)

        return mel_spec, spec
