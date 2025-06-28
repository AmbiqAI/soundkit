from typing import Type, Dict
from pydantic import BaseModel
import tensorflow as tf

# === Import model classes ===
from .SimpleFC import SimpleFC
from .crnn import CRNN, CRNNParams
from .crnn_new import crnn_new, newCRNNParams

from .ccrnn import CCRNN, CCRNNParams

from .unet import unet
from .unet_sublayers import UNetParams

# === Model Factory ===

ModelType = Type[tf.keras.Model]
class ModelFactory:
    """Factory to register and retrieve Keras model classes."""
    _registry: Dict[str, ModelType] = {}

    @classmethod
    def register(cls, name: str, model_cls: ModelType):
        cls._registry[name] = model_cls

    @classmethod
    def get(cls, name: str, **kwargs) -> tf.keras.Model:
        if name not in cls._registry:
            raise ValueError(f"Model '{name}' is not registered.")
        return cls._registry[name](**kwargs)

    @classmethod
    def list(cls) -> list[str]:
        return list(cls._registry.keys())

# === Model Parameter Factory ===
class ModelParamFactory:
    """Factory to register and retrieve Pydantic model parameter classes."""
    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, param_cls: Type[BaseModel]):
        cls._registry[name] = param_cls

    @classmethod
    def get(cls, name: str) -> Type[BaseModel]:
        if name not in cls._registry:
            raise ValueError(f"Params '{name}' are not registered.")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        return list(cls._registry.keys())


# Register models
ModelFactory.register("crnn", CRNN)
ModelFactory.register("crnn_new", crnn_new)


ModelFactory.register("ccrnn", CCRNN)
ModelFactory.register("unet", unet)

# Register parameter schemas
ModelParamFactory.register("crnn", CRNNParams)
ModelParamFactory.register("crnn_new", newCRNNParams)

ModelParamFactory.register("ccrnn", CCRNNParams)

ModelParamFactory.register("unet", UNetParams)
