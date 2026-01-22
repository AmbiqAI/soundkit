
# Real-Time Demo for Keyword Spotting (KWS)

The demo module in **soundKIT** allows real-time evaluation of your trained KWS model using either a PC microphone or Ambiq Evaluation Board (EVB). This is ideal for validating performance in live conditions.

---

## Running the Demo

To run the demo on your chosen platform:

```bash
soundkit -t kws -m demo -c configs/kws/kws.yaml demo.platform=pc # or evb
```

---

## Configuration (`demo` section of `kws.yaml`)

```yaml
demo:
  platform: pc  # Options: 'pc' or 'evb'
  epoch_loaded: best
  tflite_dir: ${job_dir}/tflite
  evb_dir: ${job_dir}/evb
  pre_gain: 1
  filename: def_nn1_nnvad
  param_struct_name: params_nn1_nnvad
```

---

## Demo Parameters
| **Parameter**       | **Description**                                                                                            |
| ------------------- | ---------------------------------------------------------------------------------------------------------- |
| `platform`          | Target device for the demo. Options: `"pc"` for host microphone or `"evb"` for embedded board.             |
| `epoch_loaded`      | Which model checkpoint to use. `"best"` typically loads the best validation-performing model.              |
| `tflite_dir`        | Directory where the exported TFLite model is stored. Used for inference on both PC and EVB.                |
| `evb_dir`           | Directory where the EVB-specific firmware files and C headers are generated.                               |
| `pre_gain`          | Gain factor applied to the incoming audio signal before processing (e.g., 1 for no change, >1 to amplify). |
| `filename`          | Base filename used when exporting the C header (e.g., `def_nn1_nnvad.h`).                                  |
| `param_struct_name` | Name of the C structure that holds model parameters (used during firmware integration).                    |

---

## Notes

- Ensure the TFLite model has been exported beforehand (see [Export](./export.md)).
- Adjust `pre_gain` to amplify or attenuate the microphone signal as needed.
- `filename` and `param_struct_name` define the naming conventions used when exporting model headers.

---

## Example

### PC
Run on the PC-based test for real-time KWS task.

```bash
soundkit -t kws -m demo -c configs/kws.yaml demo.platform=pc # run on PC
```

### EVB
Run on the PC-based test or [Ambiq's family of ultra-low power SoCs](https://ambiq.com/soc/) for real-time KWS task. You can choose on PC or EVB based on command-line overwrite

1. Type
  ```bash
  soundkit -t kws -m demo -c configs/kws.yaml demo.platform=evb # run on Ambiq EVB
  ```
1. Go to [https://ambiqai.github.io/soundkit/web-dashboards/sd-usb/](https://ambiqai.github.io/soundkit/web-dashboards/sd-usb/)
1. Follow the GUI instructions
1. If you have any connection issue or no waveform showing. See the [**Troubleshooting**](../troubleshooting.md)
