from typing import Type, Dict
from pydantic import BaseModel
import tensorflow as tf

# === Import model classes ===
from .SimpleFC import SimpleFC
from .crnn_new import CRNN, CRNNParams

from .ccrnn_new import CCRNN, CCRNNParams

from .unet import unet
from .cunet import cunet
from .unet_sublayers import UNetParams

from .crnn2d import crnn2d, CRNN2DParams
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

ModelFactory.register("ccrnn", CCRNN)
ModelFactory.register("unet", unet)
ModelFactory.register("cunet", cunet)
ModelFactory.register("crnn2d", crnn2d)


# Register parameter schemas
ModelParamFactory.register("crnn", CRNNParams)

ModelParamFactory.register("ccrnn", CCRNNParams)

ModelParamFactory.register("unet", UNetParams)

ModelParamFactory.register("crnn2d", CRNN2DParams)