# 🧪 Real-Time Demo (Speaker Verification - ID)

This page explains how to run a real-time **Speaker Verification (ID)** demo using a trained model. The demo supports both embedded (EVB) and PC-based platforms and demonstrates live speaker enrollment and identity matching.

---

## 🔧 Run `demo` Mode

```bash
soundkit -t id -m demo -c configs/id/id.yaml demo.platform=pc # or evb
```

## 🧾 Demo Parameters

| Parameter | Description |
|-----------|-------------|
| `platform` | `pc` or `evb`: run the model on desktop or Ambiq hardware |
| `epoch_loaded` | Model checkpoint to use for inference (`best`, `latest`, or a specific epoch number) |
| `tflite_dir` | Directory containing the exported `.tflite` model |
| `evb_dir` | Path to the EVB firmware project (used for build and deploy) |
| `pre_gain` | Optional gain multiplier applied before inference |
| `filename` | Output filename used for generated C header or binary |
| `param_struct_name` | C structure name for exported model parameters |
| `num_utterances_registered` | Number of utterances used per speaker for enrollment |
| `frames_vad_trigger_id` | Frame count to trigger ID matching after VAD confirms speech presence |

Example:

```yaml
demo:
  platform: pc
  epoch_loaded: best
  tflite_dir: ./soundkit/tasks/id/tflite
  evb_dir: ./soundkit/tasks/id/evb
  pre_gain: 1
  filename: def_nn2_nnid
  param_struct_name: params_def_nn2_nnid
  num_utterances_registered: 4
  frames_vad_trigger_id: 180
```

---

## 💻 Deployment Modes

## Example

### PC
Run on the PC-based test for real-time ID task.

```bash
soundkit -t id -m demo -c configs/id/id.yaml demo.platform=pc # run on PC 
```

### EVB
Run on the PC-based test or [Ambiq's family of ultra-low power SoCs](https://ambiq.com/soc/) for real-time KWS task. You can choose on PC or EVB based on command-line overwrite

1. Type 
  ```bash
  soundkit -t id -m demo -c configs/id/id.yaml demo.platform=evb # run on Ambiq EVB
  ```
1. Open the other terminal. Type

  ```bash
  soundkit -t id -m demo -c configs/id/id.yaml demo.platform=evb --view
  ```
  You should see a GUI pop out
1. Press the Button-0 on your EVB
1. Follow the GUI instructions
- If you have any connection issue or no waveform showing. See the [**Troubleshooting**](../troubleshooting.md)
---

## ✅ Output

- Real-time verification results (e.g., match/no match)
- Optional logging for scores, decision thresholds, and debug stats
- Timing benchmarks and event indicators for latency analysis

---

## 🧠 Tips

- Use `pre_gain` to adjust microphone input levels for better accuracy
- Ensure `tflite_dir` and `epoch_loaded` match the latest export
- Re-run `export` if model architecture or parameters have changed
- Adjust `num_utterances_registered` to fine-tune enrollment stability
