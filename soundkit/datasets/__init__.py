from typing import Callable, Dict

# === Import built-in + BYOD registry functions ===
from .sk_datasets import (
    load_train_clean_100,
    load_train_clean_360,
    load_dev_clean,
    load_test_clean,
    load_thchs30,
    load_wham_noise,
    load_fsd50k,
    load_esc50,
    load_musan,
    load_rirs_noises,
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
SKDatasetFactory.register("train-clean-360", load_train_clean_360)
SKDatasetFactory.register("dev-clean", load_dev_clean)
SKDatasetFactory.register("test-clean", load_test_clean)
SKDatasetFactory.register("thchs30", load_thchs30)

# Noise
SKDatasetFactory.register("wham_noise", load_wham_noise)
SKDatasetFactory.register("FSD50K", load_fsd50k)
SKDatasetFactory.register("ESC-50-master", load_esc50)
SKDatasetFactory.register("musan", load_musan)

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
