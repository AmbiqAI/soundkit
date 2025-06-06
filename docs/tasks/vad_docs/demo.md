# 🧪 Real-Time Demo (Voice Activity Detection - VAD)

This page explains how to run a real-time **Voice Activity Detection (VAD)** demo using a trained model. Demos can be executed on embedded hardware (EVB) or directly in-browser using WebUSB.

---

## 🔧 Run `demo` Mode

```bash
soundkit -t vad -m demo -c vad.yaml
```

## 🧾 Demo Parameters

| Parameter | Description |
|-----------|-------------|
| `epoch_loaded` | Model checkpoint to use for inference (`best`, `latest`, or a specific integer) |
| `tflite_dir` | Directory containing the exported `.tflite` model |
| `evb_dir` | Path to embedded board (EVB) project directory (used for firmware build/deploy) |
| `pre_gain` | Optional gain factor applied before inference (for debugging or level adjustment) |
| `filename` | Output filename for generated C header or binary |
| `param_struct_name` | C structure name to hold the exported model parameters |

Example:

```yaml
demo:
  platform: evb
  epoch_loaded: best
  tflite_dir: ./soundkit/tasks/vad/tflite
  evb_dir: ./soundkit/tasks/vad/evb
  pre_gain: 1
  filename: def_nn1_nnvad
  param_struct_name: params_nn1_nnvad
```

---

## 💻 Deployment Modes

### 🔌 Embedded Board (EVB)

- Builds firmware from `evb_dir`
- Deploys it via J-Link or USB
- Streams audio in/out through onboard codec or USB interface

Ensure:
- The EVB is connected and recognized
- Required toolchains are installed (e.g., GNU Arm, J-Link, Make)

---

### 🌐 WebUSB Demo

- Runs in **Chrome-based browsers**
- Uses `.tflite` model with in-browser audio I/O
- Requires microphone access and HTTPS or `localhost`

To launch:

1. Serve the `webusb/` directory (if included)
2. Connect the device using WebUSB
3. Load model and start inference

> ⚠️ WebUSB works only in secure contexts and modern Chromium browsers.

---

## ✅ Output

- Real-time voice activity detection results
- Optional logs for debug and latency measurement
- Can include visual indicators for speech segments

---

## 🧠 Tips

- Use `pre_gain` to normalize input amplitude
- Ensure `tflite_dir` and `epoch_loaded` match your export step
- Re-run `export` if the model structure or parameters change
