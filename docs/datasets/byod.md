# 📦 Bring Your Own Dataset (BYOD) in soundKIT

soundKIT supports a flexible BYOD (Bring Your Own Dataset) system. You can define custom datasets—speech, noise, or reverb—by adding loader functions and explicitly registering them in `soundkit/datasets/__init__.py`, similar to how models are registered.

---

## 🗂️ Directory Structure

Organize your custom datasets under:

```
soundkit/
└── soundkit/
    └── datasets/
        ├── __init__.py                 # dataset factory class and central registry for dataset loaders
        ├── sk_datasets.py              # built-in dataset loader functions
        └── your_own_registry.py        # ← your custom dataset functions
```

---

## ✍️ Example: Custom Dataset Loader

In your custom registry file (e.g., `your_own_registry.py`):

```python
# soundkit/datasets/your_own_registry.py

import os
import random
from soundkit.datasets import SKDatasetFactory

def get_wavefiles(folder: str):
    return [
        os.path.join(dp, f)
        for dp, _, fs in os.walk(folder)
        for f in fs if f.endswith(".wav")
    ]

def load_my_speech(_):
    return get_wavefiles("/path/to/my/speech")

def load_my_noise(_):
    files = get_wavefiles("/path/to/my/noise")
    random.shuffle(files)
    split = len(files) // 5
    return {"train": files[split:], "val": files[:split]}
```

Then go to `datasets/__init__.py` and register them:

```python
from .your_own_registry import load_my_speech, load_my_noise

SKDatasetFactory.register("my_speech", load_my_speech)
SKDatasetFactory.register("my_noise", load_my_noise)
```

---

## ⚙️ Using Your Dataset in Config

You can now reference your dataset in YAML configs under the `data.corpora` section.

### Example (from `se.yaml`):

```yaml
data:
  corpora:
    - {name: my_speech, type: speech, split: train}
    - {name: my_noise, type: noise, split: train-val}
    - {name: my_rir, type: reverb, split: train-val}
```

The `type` must be one of: `speech`, `noise`, or `reverb`. The `split` field defines whether the data will be used for training, validation, or both.

---

## ✅ Summary

| Action                 | What to Do                                             |
| ---------------------- | ------------------------------------------------------ |
| 📝 Add loader function | Define in a `*_registry.py` under `datasets/`          |
| 🧠 Register it         | Explicitly in `datasets/__init__.py` via `.register()` |
| 🧾 Use in config       | Reference in `data.corpora` list in YAML               |
| 🔁 No magic            | Clear, predictable, and maintainable structure         |
| 📦 Use by name         | Use in YAML as `name: my_speech`, etc.                 |

---

By following this factory-style pattern, BYOD in soundKIT becomes robust, traceable, and simple to scale.
