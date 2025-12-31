# 🧪 Real-Time Demo

This page explains how to run a real-time **Speech Enhancement (SE)** demo using a trained model. Demos can be executed on embedded hardware (EVB) or directly in-browser using WebUSB.

---

## 🔧 Run `demo` Mode

```bash
soundkit -t se -m demo -c configs/se/se.yaml demo.platform=pc # or evb
```

## 🧾 Demo Parameters

| Parameter | Description |
|-----------|-------------|
| `epoch_loaded` | Model checkpoint to use for inference (`best`, `latest`, or a specific integer) |
| `platform` | Target platform for demo execution. Options: `pc` (run on local machine) or `evb` (run on embedded board). |
| `tflite_dir` | Directory containing the exported `.tflite` model |
| `evb_dir` | Path to embedded board (EVB) project directory (used for firmware build/deploy) |
| `pre_gain` | Optional gain factor applied before inference (for debugging or level adjustment) |


Example:

```yaml
demo:
    epoch_loaded: best
    platform: pc # or evb
    tflite_dir: ./soundkit/tasks/se/tflite
    evb_dir: ./soundkit/tasks/se/evb
    pre_gain: 1
```

---

## 💻 Deployment Modes

### 🔌 PC

- Type
    ```bash
    soundkit -t se -m demo -c configs/se/se.yaml demo.platform=pc # or evb
    ```
- A GUI will pop up. Click start to demo
### 🔌 Embedded Board (EVB)
- Type
    ```bash
    soundkit -t se -m demo -c configs/se/se.yaml demo.platform=evb # or evb
    ```
- Open your browser on [nnse-usb-dashboard](https://ambiqai.github.io/web-ble-dashboards/nnse-usb/)
- Switch raw or enhance audio via pressing Button-0 on EVB
- If you have any connection issue or no waveform showing. See the [**Troubleshooting**](../troubleshooting.md)
---