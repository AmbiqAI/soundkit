# 🧪 Real-Time Demo (Voice Activity Detection - VAD)

This page explains how to run a real-time **Voice Activity Detection (VAD)** demo using a trained model. Demos can be executed on embedded hardware (EVB) or directly in-browser using WebUSB.

---

## 🔧 Run `demo` Mode

```bash
soundkit -t vad -m demo -c configs/vad/vad.yaml demo.platform=pc # or demo.platform=evb
```

## 🧾 Demo Parameters

| Parameter | Description |
|-----------|-------------|
| `platform` | `pc` or `evb`: run the model on Ambiq |
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


### 🔌 PC

The fast deploying tflite on your PC for real-time testing.

- Ensure you have a microphone connected to your PC (or built-in one)
  ```bash
  soundkit -t vad -m demo -c configs/vad/vad.yaml demo.platform=pc 
  ```
- A simple GUI will show up to start


### 🔌 Embedded Board (EVB)

1. Type:
  ```bash
  soundkit -t vad -m demo -c configs/vad/vad.yaml demo.platform=evb
  ```
1. Go to [https://ambiqai.github.io/soundkit/docs/web-dashboards/sd-usb/](https://ambiqai.github.io/soundkit/docs/web-dashboards/sd-usb/)

1. Follow the GUI instructions
1. If you have any connection issue or no waveform showing. See the [**Troubleshooting**](../troubleshooting.md)
---
