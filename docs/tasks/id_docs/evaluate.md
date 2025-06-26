# 📊 Evaluation (Speaker Verification - ID)

This page describes how to evaluate a trained **Speaker Verification (ID)** model using the `soundkit` CLI. This process runs identity verification on test audio samples and reports speaker matching results and verification metrics.

---

## 🔧 Run `evaluate` Mode

```bash
soundkit -t id -m evaluate -c id.yaml
```

---

## 🧾 Evaluation Parameters

| Parameter | Description |
|-----------|-------------|
| `epoch_loaded` | Model checkpoint to load for evaluation. Use `best`, `latest`, or a specific epoch number |
| `data.dir` | Path to the folder containing WAV files for evaluation |
| `data.files` | List of WAV filenames (relative to `data.dir`) for enrollment and testing |
| `result_folder` | Directory to save evaluation outputs, including scores, match results, and visualizations |

Example:

```yaml
evaluate:
  epoch_loaded: best
  data:
    dir: ./wavs/vad/test_wavs
    files: [rpc_audio_raw.wav, speech.wav, i_like_steak.wav, keyboard_steak.wav, steak_hairdryer.wav]
  result_folder: ./soundkit/tasks/id/test_results/crnn100_speakerid
```

---

## 📈 Output

Running the evaluation step will generate:

- Embeddings and similarity scores between enrollment and test utterances
- Decision thresholds and match/no-match outputs
- Metrics such as **Equal Error Rate (EER)**, **ROC AUC**, and **confusion matrices**
- Visuals for similarity heatmaps and threshold calibration (if configured)

> 📌 Ensure the `id.yaml` configuration matches the model architecture and feature settings used during training and export.

---

## 🛠 Advanced Tips

- Use clean and consistent enrollment audio for best verification accuracy
- Works with few-shot enrollment (e.g., 1–4 utterances per speaker)
- Supports evaluation on both open-set and closed-set verification scenarios
- Ideal for testing model robustness to channel mismatch or noise

---

Need to test the model live? See the [Demo](./demo.md) guide for PC and EVB deployment.
