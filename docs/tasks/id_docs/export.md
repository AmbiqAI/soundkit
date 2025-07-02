# 📤 Model Export (Speaker Verification - ID)

This page explains how to export a trained **Speaker Verification (ID)** model for deployment on embedded platforms or use in PC-based and WebUSB demos.

---

## 🔧 Run `export` Mode

```bash
soundkit -t id -m export -c configs/id/id.yaml
```

This command loads a trained speaker embedding model and converts it into deployable formats such as TFLite and C arrays for real-time inference.

---

## 🧾 Export Parameters

| Parameter | Description |
|-----------|-------------|
| `epoch_loaded` | Epoch of the checkpoint to export (`best`, `latest`, or a specific epoch number) |
| `tflite_dir` | Directory where exported model files will be saved (e.g., `.tflite`, `.cc`, `.h`) |

Example:

```yaml
export:
  epoch_loaded: best
  tflite_dir: ./soundkit/tasks/id/tflite
```

---

## 📦 Exported Artifacts

The export step may produce the following files depending on configuration:

| File | Description |
|------|-------------|
| `model.tflite` | Quantized TensorFlow Lite model for on-device speaker embedding |
| `model.cc`, `model.h` | C array versions of the model for use with TFLM firmware |
| `params_def_nn2_nnid.h` | Header file containing model-specific constants and metadata |
| `quant_stats.json` | (Optional) File with quantization calibration stats used during export |

---

## 🔌 Integration Targets

Exported models can be used for:

- **On-device speaker verification** with TensorFlow Lite Micro (TFLM) on Ambiq MCUs (e.g., Apollo5)
- **PC-based testing** and prototyping using TFLite runtimes
- **WebUSB demos** using in-browser `.tflite` inference
- **Firmware integration** in authentication and access control pipelines

---

## 🧠 Notes

- Make sure `epoch_loaded` matches the best model checkpoint from training
- Use the same `tflite_dir` path across `export`, `demo`, and any embedded firmware builds
- Re-run `export` if the model structure, feature settings, or sampling rate changes

For deployment instructions, see the [Demo](./demo.md) guide.
