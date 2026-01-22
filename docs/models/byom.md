
# Bring Your Own Model (BYOM)

soundKIT is designed for modularity — you can easily plug in your own model into the training, evaluation, export, and demo pipeline. This guide demonstrates how to add a custom model using `SimpleFC` as an example.

## Step 1: Define Your Model

Create a new file `SimpleFC.py` under the `models/` folder:

```python
# models/SimpleFC.py

from pydantic import BaseModel
import tensorflow as tf

class SimpleFCParams(BaseModel):
    units: int = 128
    dim_out: int = 257

class SimpleFC(tf.keras.Model):
    def __init__(self, params: SimpleFCParams = SimpleFCParams(), **kwargs):
        super().__init__()
        self.params = params
        self.fc1 = tf.keras.layers.Dense(params.units, activation='relu')
        self.fc2 = tf.keras.layers.Dense(params.units, activation='relu')
        self.out = tf.keras.layers.Dense(params.dim_out, activation='sigmoid')

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)
```

## Step 2: Register the Model

Register your model class and parameter schema in the model factory:

```python
# soundkit/models/__init__.py

from .SimpleFC import SimpleFC, SimpleFCParams
ModelFactory.register("SimpleFC", SimpleFC)
ModelParamFactory.register("SimpleFC", SimpleFCParams)
```

## Step 3: Configure your model architecture

```yaml
# soundkit/models/arch_configs/config_simplefc.yaml

name: SimpleFC
units: 256
dim_out: 257
```

## Step 4: Specify the model architecture in your YAML configuration

```yaml
# configs/se_simplefc.yaml

data:
    ...
train:
    ...
    model: ./soundkit/models/arch_configs
    config_file: config_simplefc.yaml

evaluate:
    ...

```

## Step 5: Train with Your Model

You can now train your model just like any built-in model:

```bash
soundkit -t se -m train -c configs/se_simplefc.yaml
```
