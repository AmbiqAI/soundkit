# 📊 Evaluation

This page describes how to evaluate a trained Speech Enhancement (SE) model using the `soundkit` CLI. You can compute metrics like SI-SDR, PESQ, STOI, or DNSMOS on custom WAV files or test datasets.

---

## 🔧 Run `evaluate` Mode

```bash
soundkit -t se -m evaluate -c your_config.yaml
```

---

## 🧾 Evaluation Parameters

| Parameter | Description |
|-----------|-------------|
| `epoch_loaded` | Epoch number of the model checkpoint to load for evaluation. Use `best`, `latest`, or a specific integer |
| `data.dir` | Directory containing WAV files to evaluate |
| `data.files` | List of WAV filenames (relative to `data.dir`) for evaluation |
| `result_folder` | Folder to save evaluation results (e.g., audio output, spectrograms, plots) |

Example:

```yaml
evaluate:
  epoch_loaded: best
  data:
    dir: ./wavs/se/test_wavs
    files: [keyboard_steak.wav, i_like_steak.wav, steak_hairdryer.wav]
  result_folder: ./soundkit/tasks/se/test_results/crnn100_lookahead0
```

---

## 📈 Output

Running evaluation produces:

- Enhanced audio files saved to `result_folder`
- Visualization plots (e.g., waveforms, spectrograms)
- Optional metrics like SI-SDR, PESQ, or STOI if supported

> 📌 Ensure your config references the same model and feature settings used during training.

---

## 🛠 Advanced Tips

- Evaluation supports running on multiple audio files with different noise types.
- You can evaluate using test TFRecords (if implemented in future versions) or directly with WAV files as shown.
