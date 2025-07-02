# 📊 Evaluation (Speaker Verification - ID)

This page describes how to evaluate a trained **Speaker Verification (ID)** model using the `soundkit` CLI. This process runs identity verification on test audio samples and reports speaker matching results and verification metrics.

---

## 🔧 Run `evaluate` Mode

```bash
soundkit -t id -m evaluate -c configs/id/id.yaml
```

---

## 🧾 Evaluation Parameters

| Parameter | Description |
|-----------|-------------|
| `epoch_loaded` | Model checkpoint to load for evaluation. Use `best`, `latest`, or a specific epoch number |
| `threshold_id` ( in [0,1]) | 0.8 is suggested. the threshold to determine whether the enroll person is verified
| `data.dir` | Path to the folder containing WAV files for evaluation |
| `data.reg_files` | List of WAV filenames (relative to `data.dir`) for enrollment |
| `data.test_files` | List of WAV filenames (relative to `data.dir`) for testing |

Example:

```yaml
evaluate:
  epoch_loaded: best

  threshold_id: 0.8 # threshold for id verification

  data: 
    dir: "./wavs/id/test_wavs"
    reg_files: [registration1.wav, registration2.wav, registration3.wav, registration4.wav] # list of registration files
    test_files: [test1.wav, test2.wav] # list of test files to evaluate
    
```
In the registration phase, 4 utterances for one person are recorded in **registration1-4.wav** . then NN will verify if **test1-2.wav** is belonging to this person.  
---



## 🛠 Advanced Tips

- Use clean and consistent enrollment audio for best verification accuracy
- Works with few-shot enrollment (e.g., 1–4 utterances per speaker)
- Supports evaluation on both open-set and closed-set verification scenarios
- Ideal for testing model robustness to channel mismatch or noise

---

Need to test the model live? See the [Demo](./demo.md) guide for PC and EVB deployment.
