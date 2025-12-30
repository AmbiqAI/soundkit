# Evaluation for Keyword Spotting (KWS)

The evaluation step in **SoundKit** for KWS allows you to assess model performance on specific audio samples. It visualizes keyword activity predictions and provides insight into real-world accuracy.

---

## Running Evaluation

To evaluate your trained KWS model on a set of test audio files:

```bash
soundkit -t kws -m evaluate -c configs/kws/kws.yaml
```

This will output prediction visualizations and save results to the configured folder.

---

## Configuration (`evaluate` section of `kws.yaml`)

```yaml
evaluate:
  epoch_loaded: best
  data: 
    dir: "./wavs/vad/test_wavs"
    files: [rpc_audio_raw.wav, speech.wav, i_like_steak.wav, keyboard_steak.wav, steak_hairdryer.wav]
  result_folder: ${job_dir}/test_results/${train.path.full_name}
```

---

## Evaluation Parameters

| Parameter         | Type      | Description                                                      | Example Value                |
|-------------------|-----------|------------------------------------------------------------------|------------------------------|
| `epoch_loaded`    | int/str   | Which model checkpoint to use for evaluation (`best`, `random`, `latest`, or epoch number) | `best`                       |
| `eval`            | bool      | Whether to run evaluation                                        | `true`                       |
| `data.dir`        | str       | Directory containing evaluation audio files                      | `./wavs/kws/test_wavs`       |
| `data.files`      | list[str] | List of audio files to evaluate                                  | `[speech.wav, noise.wav]`     |
| `data.result_folder` | str    | Where to save evaluation results                                 | `./test_results/kws_model`   |

### Example YAML
```yaml
evaluate:
  epoch_loaded: best
  eval: true
  data:
    dir: "./wavs/kws/test_wavs"
    files: [speech.wav, noise.wav]
    result_folder: ./test_results/kws_model
```

---

## Notes

- **epoch_loaded**: Specifies which checkpoint to load (`best` or specific epoch number).
- **data.files**: Select test files to run keyword detection inference on.
- **result_folder**: Output location for saving plots and analysis results.

---

## Tips

- Ensure you have trained the model before evaluating.
- You can customize the `data.dir` and `files` to match your real-world use cases.
- Results include frame-wise detection output and keyword activity plots.

Refer to the [Training Guide](./train.md) if you haven't yet trained your model.
