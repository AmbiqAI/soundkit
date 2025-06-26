# 🧪 Real-Time Demo (Speaker Verification - ID)

This page explains how to run a real-time **Speaker Verification (ID)** demo using a trained model. The demo supports both embedded (EVB) and PC-based platforms and demonstrates live speaker enrollment and identity matching.

---

## 🔧 Run `demo` Mode

```bash
soundkit -t id -m demo -c id.yaml
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

### 🔌 Embedded Board (EVB)

- Builds and flashes firmware from `evb_dir`
- Streams audio via onboard codec or USB mic
- Performs enrollment and verification on-device
- Optionally interfaces with host PC for result display

**Requirements:**

- Ambiq EVB connected and recognized
- Toolchains installed (e.g., GNU Arm, J-Link, Make)

---

### 🖥️ PC-Based Demo

- Runs entirely on your desktop using a connected microphone
- Uses `.tflite` model for inference via TFLite runtime
- Performs local speaker registration and verification

To run:

1. Export your model using the `export` mode
2. Launch the `demo` with `platform: pc`
3. Speak to register, then verify against stored embeddings

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
