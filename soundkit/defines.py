""""SoundKit configuration definitions."""
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import tempfile
from pathlib import Path
# Backport StrEnum for Python < 3.11
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    class StrEnum(str, Enum):
        pass

@dataclass
class SignalConfig:
    """Signal configuration for data processing."""
    sampling_rate: int = 16000
    dc_removal: bool = True

@dataclass
class DataConfig:
    """Data configuration for training and evaluation."""
    path_tfrecord: str = ""
    tfrecord_datalist_name: Dict[str, str] = field(default_factory=dict)
    num_samples_per_noise: Any = None
    force_download: bool = False
    reverb_prob: float = 0.0
    num_processes: int = 1
    corpora: List[Dict[str, Any]] = field(default_factory=list)
    snr_dbs: List[int] = field(default_factory=list)
    target_length_in_secs: float = 5.0
    min_amp: float = 0.05
    max_amp: float = 0.95
    signal: SignalConfig = field(default_factory=SignalConfig)
    target_frames_extension: int = 10 # for kws
    num_sentences: int = 10  # for id
    ppls_per_group: int = 8  # for id
    debug: bool = False

    def __post_init__(self):
        if self.num_processes <= 0:
            raise ValueError("num_processes must be > 0")

        if not (0.0 <= self.reverb_prob <= 1.0):
            raise ValueError(
                "reverb_prob must be between 0.0 and 1.0")

        if self.target_length_in_secs <= 0.0:
            raise ValueError("target_length_in_secs must be > 0.0")

        if not (0.0 <= self.min_amp < self.max_amp <= 1.0):
            raise ValueError(
                "min_amp and max_amp must satisfy 0.0 <= min_amp < max_amp <= 1.0")

        # required_keys = {"train", "val"}
        # if self.num_samples_per_noise is not None:
        #     if not required_keys.issubset(self.num_samples_per_noise):
        #         missing = required_keys - self.num_samples_per_noise.keys()
        #         raise ValueError(
        #             f"num_samples_per_noise must have keys {required_keys}, missing: {missing}")
@dataclass
class LossFunctionConfig:
    """Configuration for loss function."""
    type: str = "compressed_mse"
    params: Dict[str, Any] = field(default_factory=lambda: {"exp": 0.6, "eps": 1e-8})
    def __post_init__(self):
        allowed_types = {
            "mse",
            "mae",
            "compressed_mse",
            "mrl_mse",
            "log_mse",
            "focal",
            "cross_entropy",
            "si_sdr",
            "compound_loss",
            "time_smooth_mae"
        }
        if self.type not in allowed_types:
            raise ValueError(
                f"Loss function type must be one of {allowed_types}, got '{self.type}'"
            )
@dataclass
class PathConfig:
    """Path configuration for training."""
    full_name: str = ""
    checkpoint_dir: str = ""
    tensorboard_dir: str = ""

@dataclass
class FeatureConfig:
    """Feature extraction parameters."""
    frame_size: int = 480
    hop_size: int = 160
    fft_size: int = 512
    type: str = "mel"
    bins: int = 72
    bins_fft: int = 50
    n_mels: int = 32
    def __post_init__(self):
        allowed_types = {
            "spec",
            "pspec",
            "logpspec",
            "logampspec",
            "mel",
            "hybrid",
            "hybrid_mag",
            "time",
            "erb_complex",
            "erb_mag",
        }
        if self.type not in allowed_types:
            raise ValueError(
                f"type must be one of {allowed_types}, got '{self.type}'"
            )
        # Enforce bins_fft and n_mels for hybrid type
        if self.type == "hybrid":
            if self.bins_fft <= 0 or self.n_mels <= 0:
                raise ValueError(
                    "For type 'hybrid', both bins_fft and n_mels must be set and non-zero.")

        if self.type == "mel":
            if self.bins <= 0:
                raise ValueError("For type 'mel', bins must be set and non-zero.")

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    config_dir: str = ""
    config_file: str = ""
    override: Any = None
@dataclass
class TrainConfig:
    """Training configuration parameters."""
    initial_lr: float = 0.0
    num_per_epoch_files: Dict[str, Optional[int]] = field(
        default_factory=lambda: {"train": None, "val": None}
    )
    truncate_time: Optional[float] = None
    lr_schedule: str = "cosine"
    batchsize: int = 8
    spec_aug: bool = False
    epochs: int = 10
    warmup_epochs: int = 0
    epoch_loaded: Any = "random"
    reset_states_every_batch: bool = True
    loss_function: LossFunctionConfig = field(default_factory=LossFunctionConfig)
    path: PathConfig = field(default_factory=PathConfig)
    num_lookahead: int = 0
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    standardization: bool = False
    standardization_type: str = "mve"
    model: ModelConfig = field(default_factory=ModelConfig)
    debug: bool = False
    def __post_init__(self):
        if self.initial_lr < 0.0:
            raise ValueError("initial_lr must be >= 0")
        if self.num_lookahead < 0:
            raise ValueError("num_lookahead must be >= 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.batchsize <= 0:
            raise ValueError("batchsize must be > 0")
        if self.lr_schedule not in ("cosine", "constant"):
            raise ValueError("lr_schedule must be 'cosine' or 'constant'")
        if self.standardization_type not in ("mve", "mean"):
            raise ValueError("standardization_type must be 'mve' or 'mean'")
        # Check epoch_loaded
        if isinstance(self.epoch_loaded, int) and self.epoch_loaded < 0:
            raise ValueError(
                "epoch_loaded must be >= 0 if it is an int")
        if isinstance(self.epoch_loaded, str) and self.epoch_loaded not in ("best", "random", "latest"):
            raise ValueError(
                "epoch_loaded must be 'best', 'random', 'latest', or a non-negative integer")

@dataclass
class EvaluateDataConfig:
    """Data configuration for evaluation."""
    dir: str = ""
    files: List[str] = field(default_factory=list)
    reg_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    result_folder: str = ""

@dataclass
class EvaluateConfig:
    """Evaluation configuration parameters."""
    epoch_loaded: Any = "random"
    eval: bool = False
    threshold_id: float = 0.8  # threshold for id verification
    data: EvaluateDataConfig = field(default_factory=EvaluateDataConfig)
    # Check epoch_loaded
    def __post_init__(self):
        if isinstance(self.epoch_loaded, int) and self.epoch_loaded < 0:
            raise ValueError(
                "epoch_loaded must be >= 0 if it is an int")
        if isinstance(self.epoch_loaded, str) and self.epoch_loaded not in ("best", "random", "latest"):
            raise ValueError(
                "epoch_loaded must be 'best', 'random', 'latest', or a non-negative integer")

@dataclass
class ExportConfig:
    """Export configuration parameters."""
    dtype: str = ""
    eval: bool = False
    qbit_input: int = 8
    calibration_samples: int | None = None
    epoch_loaded: Any = "random"
    tflite_dir: str = ""
    # Check epoch_loaded
    def __post_init__(self):
        if isinstance(self.epoch_loaded, int) and self.epoch_loaded < 0:
            raise ValueError(
                "epoch_loaded must be >= 0 if it is an int")
        if isinstance(self.epoch_loaded, str) and self.epoch_loaded not in ("best", "random", "latest"):
            raise ValueError(
                "epoch_loaded must be 'best', 'random', 'latest', or a non-negative integer")

@dataclass
class DemoConfig:
    """Demo configuration parameters."""
    epoch_loaded: Any = "random"
    platform: str = "pc"
    tflite_dir: str = ""
    evb_dir: str = ""
    pre_gain: float = 1.0
    filename: str = ""
    param_struct_name: str = "model_params"
    dtype: str = "int16"
    nbits: int = 16
    qbits: int = 8
    calibration_samples: int | None = None
    num_utterances_registered: int = 4 # number of registered utterances for VAD ID
    frames_vad_trigger_id: int = 180 # number of frames to trigger VAD for ID
    # Check epoch_loaded
    def __post_init__(self):
        if isinstance(self.epoch_loaded, int) and self.epoch_loaded < 0:
            raise ValueError(
                "epoch_loaded must be >= 0 if it is an int")
        if isinstance(self.epoch_loaded, str) and self.epoch_loaded not in ("best", "random", "latest"):
            raise ValueError(
                "epoch_loaded must be 'best', 'random', 'latest', or a non-negative integer")
        if self.platform not in ("pc", "evb"):
            raise ValueError(
                "platform must be 'pc' or 'evb'")

@dataclass
class SKTaskParams:
    """SoundKit task parameters."""
    name: str = "experiment"
    project: str = "soundkit"
    job_dir: Path = field(default_factory=tempfile.gettempdir)

    # Task-specific arguments
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)

class SKMode(StrEnum):
    """SoundKit execution modes."""
    data: str = "data"
    train: str = "train"
    evaluate: str = "evaluate"
    export: str = "export"
    demo: str = "demo"
