
# Keyword Spotting (KWS)

The [Keyword Spotting (KWS)](./kws_docs/introduction.md) module in **soundKIT** enables accurate detection of predefined spoken keywords in diverse and noisy conditions. It is designed for real-time, low-power deployments using embedded platforms.

It supports:

* ✅ [Data preparation](./kws_docs/data.md)
* ✅ [Model training](./kws_docs/train.md)
* ✅ [Evaluation](./kws_docs/evaluate.md)
* ✅ [Model export](./kws_docs/export.md)
* ✅ [PC/EVB-based demo inference](./kws_docs/demo.md)

Engineered for deployment on [Ambiq's ultra-low power SoCs](https://ambiq.com/soc/), KWS offers high accuracy even in constrained edge environments.

📘 **Try it now:** Explore the [KWS Tutorial Notebook](/notebooks/SoundKit_KWS_Tutorial/) to get started.

---

## Features

- Keyword spotting with frame-level granularity
- CRNN-based acoustic modeling
- Support for long audio (15s) with extension frames
- TFLite and C-array export for embedded use
- PC-based and EVB-based real-time demos

---

## Install soundKIT

Refer to the [QuickStart Guide](../quickstart.md) for installation and setup instructions.

---

## KWS Task Modes

Use the `soundkit` CLI to run KWS tasks through different modes using a configuration file like `kws.yaml`.

??? example "`kws.yaml`"
    ```yaml
    name: galaxy
    project: kws
    job_dir: ./soundkit/tasks/${project}
    ...
    ```

!!! note "KWS Task Mode Overview"

    === "Data"
        Prepares training and validation samples by injecting keywords into a diverse background of speech and noise at different SNR levels.

        ```bash
        soundkit -t kws -m data -c configs/kws/kws.yaml
        ```

        See [Data](./kws_docs/data.md) for dataset setup details.

    === "Train"
        Trains a keyword spotting model with CRNN architecture, focal loss, and SNR-augmented training.

        ```bash
        soundkit -t kws -m train -c configs/kws/kws.yaml
        ```

        Optional: run TensorBoard for live monitoring:

        ```bash
        soundkit -t kws -m train --tensorboard -c configs/kws/kws.yaml
        ```

        See [Train](./kws_docs/train.md) for training methodology.

    === "Evaluate"
        Evaluates model performance on sample audio files, producing keyword activity predictions.

        ```bash
        soundkit -t kws -m evaluate -c configs/kws/kws.yaml
        ```

        See [Evaluate](./kws_docs/evaluate.md) for usage details.

    === "Export"
        Converts the trained model into TFLite and C-format for deployment on Ambiq EVB or other embedded targets.

        ```bash
        soundkit -t kws -m export -c configs/kws/kws.yaml
        ```

        See [Export](./kws_docs/export.md) for model format options.

    === "Demo"
        Test the model in real-time either using PC or EVB hardware. We suggest to run on your PC first and try on EVB later:

        ```bash
        soundkit -t kws -m demo -c configs/kws/kws.yaml demo.platform=pc
        # or
        soundkit -t kws -m demo -c configs/kws/kws.yaml demo.platform=evb

        ```

        See [Demo](./kws_docs/demo.md) to test live detection.
