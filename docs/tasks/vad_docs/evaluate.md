# 📊 Evaluation (Voice Activity Detection - VAD)

This page describes how to evaluate a trained Voice Activity Detection (VAD) model using the `soundkit` CLI. You can run inference on custom WAV files and generate voice activity predictions.

---

## 🔧 Run `evaluate` Mode

```bash
soundkit -t vad -m evaluate -c vad.yaml
```

---

## 🧾 Evaluation Parameters

| Parameter | Description |
|-----------|-------------|
| `epoch_loaded` | Model checkpoint to load for evaluation. Use `best`, `latest`, or a specific integer |
| `data.dir` | Path to the folder containing WAV files for evaluation |
| `data.files` | List of WAV filenames (relative to `data.dir`) to evaluate |
| `result_folder` | Directory to save prediction results, plots, and related outputs |

Example:

```yaml
evaluate:
  epoch_loaded: best
  data:
    dir: ./wavs/vad/test_wavs
    files: [rpc_audio_raw.wav, speech.wav, i_like_steak.wav, keyboard_steak.wav, steak_hairdryer.wav]
  result_folder: ./soundkit/tasks/vad/test_results/crnn100_lookahead0
```

---

## 📈 Output

Running the evaluation step will generate:

- Prediction results showing voice activity regions
- Visualization of audio signals and detected speech segments
- Annotated spectrograms or framewise outputs (if enabled in the pipeline)

> 📌 Be sure the `vad.yaml` configuration aligns with the model settings used during training and export.

---

## 🛠 Advanced Tips

- Works well on short or long-form audio files (up to the configured `target_length_in_secs`)
- Combine with `export` mode if you want to visualize `.tflite` model performance
- Great for testing behavior across various noise conditions
