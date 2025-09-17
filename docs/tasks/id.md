# Speaker Verification (ID)

The [Speaker Verification (ID)](./id_docs/introduction.md) module in **SoundKit** enables robust, on-device voice-based identity recognition. Designed for low-power edge devices, this module verifies whether a given voice sample matches a registered speaker.

It supports:

* ✅ [Data preparation](./id_docs/data.md)  
* ✅ [Model training](./id_docs/train.md)  
* ✅ [Evaluation](./id_docs/evaluate.md)  
* ✅ [Model export](./id_docs/export.md)  
* ✅ [PC/EVB-based demo inference](./id_docs/demo.md)

Built for deployment on [Ambiq’s ultra-low power MCUs](https://ambiq.com/soc/), the ID module ensures fast, secure, and private authentication directly on-device.

📘 **Try it now:** Explore the [ID Tutorial Notebook](../../notebooks/SoundKit_ID_Tutorial.ipynb) to get started.

---

## Features

- Voiceprint-based speaker identity verification  
- ResNet-style embedding model with contrastive loss  
- Enrollment and matching modes  
- TFLite and C-array export for embedded inference  
- PC-based and EVB-based real-time demo pipelines

---

## Install SoundKit

Refer to the [QuickStart Guide](../quickstart.md) for installation and setup instructions.

---

## ID Task Modes

Use the `soundkit` CLI to run ID tasks through different modes using a configuration file like `id.yaml`.

??? example "`id.yaml`"
    ```yaml
    name: galaxy_id
    project: id
    job_dir: ./soundkit/tasks/${project}
    ...
    ```

!!! note "ID Task Mode Overview"

    === "Data"
        Prepares utterances from multiple speakers with augmentations for training the speaker embedding network.

        ```bash
        soundkit -t id -m data -c configs/id/id.yaml
        ```

        See [Data](./id_docs/data.md) for dataset setup.

    === "Train"
        Trains a speaker verification model using contrastive learning and embedding normalization.

        ```bash
        soundkit -t id -m train -c configs/id/id.yaml
        ```

        Optional: run TensorBoard for live training metrics:

        ```bash
        soundkit -t id -m train --tensorboard -c configs/id/id.yaml
        ```

        See [Train](./id_docs/train.md) for training details.

    === "Evaluate"
        Measures model accuracy

        ```bash
        soundkit -t id -m evaluate -c configs/id/id.yaml
        ```

        See [Evaluate](./id_docs/evaluate.md) for evaluation metrics.

    === "Export"
        Converts the trained model to TFLite and C formats for embedded use.

        ```bash
        soundkit -t id -m export -c configs/id/id.yaml
        ```

        See [Export](./id_docs/export.md) for export options.

    === "Demo"
        Runs real-time speaker enrollment and verification demo on PC or EVB using the exported model.
        We suggest to try on PC first for fast evaluation and then EVB.

        ```bash
        soundkit -t id -m demo -c configs/id.yaml demo.platform=pc
        # or 
        soundkit -t id -m demo -c configs/id.yaml demo.platform=evb
        ```

        See [Demo](./id_docs/demo.md) to test speaker recognition live.
