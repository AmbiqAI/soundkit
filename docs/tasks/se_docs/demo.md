# 🧪 Real-Time Demo

This page explains how to run a real-time **Speech Enhancement (SE)** demo using a trained model. Demos can be executed on embedded hardware (EVB) or directly in-browser using WebUSB.

---

## 🔧 Run `demo` Mode

```bash
soundkit -t se -m demo -c your_config.yaml
```

## 🧾 Demo Parameters

| Parameter | Description |
|-----------|-------------|
| `epoch_loaded` | Model checkpoint to use for inference (`best`, `latest`, or a specific integer) |
| `tflite_dir` | Directory containing the exported `.tflite` model |
| `evb_dir` | Path to embedded board (EVB) project directory (used for firmware build/deploy) |
| `pre_gain` | Optional gain factor applied before inference (for debugging or level adjustment) |

Example:

```yaml
demo:
epoch_loaded: best
tflite_dir: ./soundkit/tasks/se/tflite
evb_dir: ./soundkit/tasks/se/evb
pre_gain: 1
```

---

## 💻 Deployment Modes

### 🔌 Embedded Board (EVB)

- Builds firmware from `evb_dir`
- Deploys it via J-Link or USB
- Streams audio in/out through onboard codec or USB interface

Make sure:
- Your board is connected and drivers are installed
- You have required toolchains (e.g., GNU Arm, J-Link, Make)

---

### 🌐 WebUSB Demo

- Works in **Chrome-based browsers** only
- Uses the `.tflite` model and in-browser audio I/O
- Requires microphone permission

To launch:

1. Serve the `webusb/` directory (if applicable)
2. Connect device via WebUSB
3. Select model and run demo

> ⚠️ WebUSB requires HTTPS or `localhost` and a compatible browser.

---

## ✅ Output

- Real-time audio enhancement
- Optional debug logs, latency measurement, and visualization (if supported)

---

## 🧠 Tips

- Use `pre_gain` to boost or attenuate input volume if needed
- Match `tflite_dir` and `epoch_loaded` with your exported model
- Re-export if changes are made to model architecture or parameters
