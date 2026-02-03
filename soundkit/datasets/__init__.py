from typing import Callable, Dict

# === Import built-in + BYOD registry functions ===
from .sk_datasets import (
    get_corpus_path,
    corpus_exists,
    load_train_clean_100,
    load_vad_train_clean_100, # for vad
    load_vad_train_clean_360, # for vad
    load_vad_train_other_500, # for vad
    load_vad_dev_clean,       # for vad
    load_vad_thchs30,           # for vad
    load_train_galaxy, # for kws
    load_val_galaxy,   # for kws
    load_train_coros,  # for kws
    load_val_coros,    # for kws
    load_train_clean_360,
    load_dev_clean,
    load_test_clean,
    load_thchs30,
    load_wham_noise,
    load_dns_challenge_train,
    load_dns_challenge_val,
    load_dns_challenge_noise,
    load_wind_noise,
    load_fsd50k,
    load_esc50,
    load_musan,
    load_rirs_noises,
    load_all_noises,
)


# === Dataset Factory ===

class SKDatasetFactory:
    """Factory for registering and retrieving dataset loaders."""
    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, loader_fn: Callable):
        cls._registry[name] = loader_fn

    @classmethod
    def get(cls, name: str) -> Callable:
        if name not in cls._registry:
            raise ValueError(f"Dataset '{name}' is not registered.")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        return list(cls._registry.keys())


# === Register datasets ===

# Speech
SKDatasetFactory.register("train-clean-100", load_train_clean_100)

SKDatasetFactory.register("vad_train-clean-100", load_vad_train_clean_100)
SKDatasetFactory.register("vad_train-clean-360", load_vad_train_clean_360)
SKDatasetFactory.register("vad_train-other-500", load_vad_train_other_500)
SKDatasetFactory.register("vad_dev-clean", load_vad_dev_clean)
SKDatasetFactory.register("vad_thchs30", load_vad_thchs30)

SKDatasetFactory.register("train-clean-360", load_train_clean_360)
SKDatasetFactory.register("dev-clean", load_dev_clean)
SKDatasetFactory.register("test-clean", load_test_clean)
SKDatasetFactory.register("thchs30", load_thchs30)

SKDatasetFactory.register("train-galaxy", load_train_galaxy)
SKDatasetFactory.register("val-galaxy", load_val_galaxy)
SKDatasetFactory.register("train-coros", load_train_coros)
SKDatasetFactory.register("val-coros", load_val_coros)
# Noise
SKDatasetFactory.register("wham_noise", load_wham_noise)

SKDatasetFactory.register("dns_challenge_train", load_dns_challenge_train)
SKDatasetFactory.register("dns_challenge_val", load_dns_challenge_val)
SKDatasetFactory.register("dns_challenge_noise", load_dns_challenge_noise)


SKDatasetFactory.register("FSD50K", load_fsd50k)
SKDatasetFactory.register("ESC-50-master", load_esc50)
SKDatasetFactory.register("musan", load_musan)
SKDatasetFactory.register("wind_noise", load_wind_noise)
SKDatasetFactory.register("all_noises", load_all_noises)
# Reverb
SKDatasetFactory.register("RIRS_NOISES", load_rirs_noises)
SKDatasetFactory.register("rirs_noises", load_rirs_noises)

# Custom BYOD


# from .your_own_registry import (
#     load_my_custom_speech,
#     load_my_custom_noise,
#     load_my_custom_rir,
# )

# SKDatasetFactory.register("my_speech", load_my_custom_speech)
# SKDatasetFactory.register("my_noise", load_my_custom_noise)
# SKDatasetFactory.register("my_rir", load_my_custom_rir)
