# 📤 Model Export (Voice Activity Detection - VAD)

This page explains how to export a trained Voice Activity Detection (VAD) model for deployment on embedded systems or in-browser inference using WebUSB.

---

## 🔧 Run `export` Mode

```bash
soundkit -t vad -m export -c configs/vad/vad.yaml
```

This command loads a trained model checkpoint and converts it into deployable formats like TFLite or embedded C source files.

---

## 🧾 Export Parameters

| Parameter | Description |
|-----------|-------------|
| `epoch_loaded` | Epoch of the checkpoint to export (`best`, `latest`, or a specific integer) |
| `tflite_dir` | Directory to store exported files (e.g., `.tflite`, `.cc`, `.h`) |

Example:

```yaml
export:
  epoch_loaded: best
  tflite_dir: ./soundkit/tasks/vad/tflite
```

---

## 📦 Exported Artifacts

Depending on your configuration, the following files may be generated:

| File | Description |
|------|-------------|
| `model.tflite` | TensorFlow Lite model optimized for MCU deployment |
| `model.cc`, `model.h` | C array representations of the TFLite model |
| `params_nn1_nnvad.h` | Header file containing model parameters for firmware integration |
| `quant_stats.json` | (Optional) JSON file with quantization statistics if quantized export is used |

---

## 🔌 Integration Targets

Exported models are suitable for:

- TensorFlow Lite Micro (TFLM) deployment on embedded processors (e.g., Ambiq, STM32)
- Voice activity detection in custom firmware/DSP applications
- WebUSB demos using `.tflite` models in-browser

---

## 🧠 Notes

- Be sure your export and demo modes use the same `epoch_loaded`
- Use `tflite_dir` consistently across `export`, `demo`, and any firmware build scripts
- If model changes (e.g., architecture, sampling rate), re-export is necessary
