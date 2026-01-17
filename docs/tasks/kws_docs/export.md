
# Model Export for Keyword Spotting (KWS)

The export step in **SoundKit** converts your trained KWS model into formats optimized for embedded deployment, including TensorFlow Lite (TFLite) and C header files for use with Ambiq SDKs.

---

## Running Export

To export the best-performing model checkpoint:

```bash
soundkit -t kws -m export -c configs/kws/kws.yaml
```

---

## Configuration (`export` section of `kws.yaml`)

```yaml
export:
  epoch_loaded: best
  tflite_dir: ${job_dir}/tflite
```

---

## Output Formats

- **TFLite (.tflite)**: For use in TensorFlow Lite runtimes or mobile applications.
- **C Header (.h)**: For direct integration into embedded C applications, particularly on Ambiq EVB platforms.

Exported files will be saved under the specified `tflite_dir`.

---

## Notes

- Make sure your model has been trained and saved before running this step.
- The `epoch_loaded` should match the best or final checkpoint you intend to deploy.
- The exported files can be directly used with the EVB or PC demos.

Refer to the [Demo Guide](./demo.md) to run real-time keyword detection using the exported model.
