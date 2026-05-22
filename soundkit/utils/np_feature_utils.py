import numpy as np
from .np_stft import StreamingSTFT, StreamingISTFT

from soundkit.utils.mel import gen_mel_bank
from soundkit.utils.mel_spec_gen import melspec_gen
from soundkit.utils.erb import ERB
from soundkit.utils.converter_fix_point import fakefix

from soundkit.utils.np_complex_utils import (
    complex_magnitude,
    complex_angle,
    polar_to_complex,
    get_compressed_complex,
)

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
            exp_complex=1.0,
            eps=1e-7,
            scale=1,
            sampling_rate=16000,
            mel_bins=72,
            thresh_mel=50,
            stream=True,
            platform="tensorflow",
            erb_subband_1=65,
            erb_subband_2=64):
        bypass_stft=False
        if feat_type =="time":
            bypass_stft=True
        self.stft_exec = StreamingSTFT(
            frame_len,
            hop_len,
            fft_len,
            stream=stream,
            bypass_stft=bypass_stft
        )
        self.exp_complex = exp_complex
        self.scale = scale
        self.sampling_rate = sampling_rate
        self.feat_type=feat_type
        self.eps = eps
        self._extractors = {
            "spec": self._extract_spec,
            "pspec": self._extract_pspec,
            "logpspec": self._extract_logpspec,
            "logampspec": self._extract_logampspec,
            "mel": self._extract_mel,
            "hybrid": self._extract_logpspec_mel,
            "time": self._extract_time,
            "erb_complex": self._extract_erb_complex, # ERB complex spectrogram
            "erb_mag": self._extract_erb_mag, # ERB magnitude spectrogram
        }

        if feat_type == "mel":
            fbanks = gen_mel_bank(
                    fftsize         = fft_len,
                    nfilt           = mel_bins,
                    sample_rate     = sampling_rate,)
            self.mel_filter = fbanks.T
            self.dim_feat = mel_bins
        elif feat_type == "hybrid":
            if fft_len is None:
                raise ValueError("bins_fft must be specified for hybrid feature extraction")
            if mel_bins is None:
                raise ValueError("n_mels must be specified for hybrid feature extraction")

            fbanks = melspec_gen(
                samplingRate=sampling_rate,
                n_fft=fft_len,
                n_mels=mel_bins,
                thresh_mel=thresh_mel)
            self.mel_filter = fbanks.T
            self.dim_feat = fbanks.shape[0]

        elif feat_type == "time":
            self.dim_feat = frame_len
        elif feat_type== "erb_complex":
            self.erb = ERB(
                erb_subband_1=erb_subband_1,
                erb_subband_2=erb_subband_2,
                nfft=fft_len,
                platform=platform)
            self.dim_feat = erb_subband_1 + erb_subband_2
        elif feat_type == "erb_mag":
            self.erb = ERB(
                erb_subband_1=erb_subband_1,
                erb_subband_2=erb_subband_2,
                nfft=fft_len,
                platform=platform)
            self.dim_feat = erb_subband_1 + erb_subband_2
        else:
            dim_feat = (fft_len // 2) + 1

            self.mel_filter = np.eye(
                dim_feat)
            self.dim_feat = dim_feat

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

    def _extract_spec(
            self,
            audio_sn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        spec = self.stft_exec.process(audio_sn)

        return spec, spec.copy()

    def _extract_erb_complex(
            self,
            audio_sn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        # 1. Get STFT
        spec = self.stft_exec.process(audio_sn)
        spec_real = np.real(spec)
        spec_imag = np.imag(spec)

        # 2. Prepare for ERB bank (matching your TF stack axis)
        spec_combined = np.stack([spec_real, spec_imag], axis=0)

        # 4. Apply Power Compression (if exp_complex != 1.0)
        
        if 1:
            if self.exp_complex != 1.0:

                spec_combined = np.sign(spec_combined) * (np.abs(spec_combined)**self.exp_complex)

        erb_spec = self.erb.bm(spec_combined)

        # 3. Reconstruct complex ERB from the BM output
        # Based on your TF code: erb_spec[...,0] was real, erb_spec[...,1] was imag
        # If self.erb.bm returns [2, T, F], then:
        real = erb_spec[0]
        imag = erb_spec[1]
        erb_complex = real + 1j * imag
        if 0:
            if self.exp_complex != 1.0:
                erb_complex = get_compressed_complex(
                    erb_complex, self.exp_complex, self.eps)

            # eps = 1e-8
            # mag = np.sqrt(np.abs(erb_complex)**2 + eps**2)**self.exp_complex
            # # Using 2**-15 to match your TF epsilon/small constant
            # phase = np.angle(erb_complex + eps)
            # erb_complex = mag * np.exp(1j * phase)

        return erb_complex, spec.copy()
    
    def _extract_erb_mag(
            self,
            audio_sn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        spec = self.stft_exec.process(audio_sn)
        erb_mag = self.erb.bm(np.abs(spec))
        return erb_mag, spec.copy()

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
    
    def _extract_time(
            self,
            audio_sn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        frames = self.stft_exec.process(audio_sn)
        spec = np.fft.rfft(frames, n=self.stft_exec.fft_len)
        return frames, spec
