# 📤 Model Export

This page explains how to export a trained Speech Enhancement (SE) model for deployment on embedded systems or in-browser inference (WebUSB).

---

## 🔧 Run `export` Mode

```bash
soundkit -t se -m export -c your_config.yaml
```

This command loads the trained model from a checkpoint and converts it into deployment-ready formats such as TFLite or C arrays.

---

## 🧾 Export Parameters

| Parameter | Description |
|-----------|-------------|
| `epoch_loaded` | Epoch number of the checkpoint to export (`best`, `latest`, or an integer) |
| `tflite_dir` | Output directory to save exported files (e.g., `.tflite`, `.cc`, `.h`) |

Example:

```yaml
export:
  epoch_loaded: best
  tflite_dir: ./soundkit/tasks/se/tflite
```

---

## 📦 Exported Artifacts

Depending on your configuration and post-processing, the following files may be produced:

| File | Purpose |
|------|---------|
| `model.tflite` | TensorFlow Lite model for embedded inference |
| `model.cc`, `model.h` | C array versions of the model weights and structure |
| `params_se.h` | Model and signal parameters (e.g., FFT size, sampling rate) for C deployment |
| `quant_stats.json` | Optional quantization statistics (if quantization is applied) |

---

## 🔌 Integration Targets

Exported models can be integrated with:

- TensorFlow Lite Micro (TFLM) on MCU targets (e.g., Ambiq Apollo)
- Custom DSP pipelines
- In-browser inference via WebUSB using `.tflite` and JavaScript

---

## 🧠 Notes

- Ensure your `model_dir` and `tflite_dir` paths are consistent.
- If exporting for a demo, make sure the same `epoch_loaded` is used in `demo` mode.
