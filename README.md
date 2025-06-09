
<p align="center">
  <a href="https://github.com/AmbiqAI/soundkit"><img src="./docs/assets/soundkit-banner.png" alt="SoundKit"></a>
</p>

---

**Documentation**: <a href="https://ambiqai.github.io/soundkit" target="_blank">https://ambiqai.github.io/soundkit</a>  
**Source Code**: <a href="https://github.com/AmbiqAI/soundkit" target="_blank">https://github.com/AmbiqAI/soundkit</a>

---

**SoundKit** is an AI Development Kit (ADK) designed to help developers build, train, and deploy real-time **audio classification** models onto [Ambiq's family of ultra-low power SoCs](https://ambiq.com/soc/). The kit includes task-specific datasets, energy-efficient model architectures, and built-in tools for optimization and deployment. Developers can use pre-trained models or create custom audio models tailored to their specific edge application.

**Key Features:**

* **Real-time**: Run low-latency inference on embedded edge devices.
* **Efficient**: Built for Ambiq’s ultra low-power hardware platforms.
* **Customizable**: Add new models, datasets, and audio tasks.
* **End-to-End**: Includes tools for training, quantization, evaluation, and deployment.
* **Open Source**: Available for use and contributions on GitHub.

---

## <span class="sk-h2-span">Requirements</span>

* [Python ^3.10+](https://www.python.org)
* [uv ^1.6.1+](https://docs.astral.sh/uv/getting-started/installation/)

The following are also required to compile/flash binaries for EVB demos:

* [Arm GNU Toolchain ^12.2](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads)
* [Segger J-Link ^7.92](https://www.segger.com/downloads/jlink/)

!!! note
    A [VSCode Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) is defined in [./.devcontainer](https://github.com/AmbiqAI/soundkit/tree/main/.devcontainer).

---

## <span class="sk-h2-span">Installation</span>

Install the `soundkit` package directly from PyPI:

```bash
pip install soundkit
```

Or install from source:

```bash
git clone https://github.com/AmbiqAI/soundkit.git
cd soundkit
pip install -e .
```

---

## <span class="sk-h2-span">Usage</span>

SoundKit can be used via CLI or directly as a Python package. It supports a flexible configuration-based workflow to streamline training and deployment.

Refer to the [Quickstart Guide](https://ambiqai.github.io/soundkit/quickstart/) to get started quickly.

---

## <span class="sk-h2-span">Tasks</span>

SoundKit supports three core **audio tasks**, each with reference pipelines for training, evaluation, export, and deployment:

- **SE (Sound Enhancement)**: Speech enhancemnt.
- **VAD (Voice Activity Detection)**: Detect presence or absence of human voice in audio streams.
- **KWS (Keyword Spotting)**: Recognize short spoken keywords, such as wake words.

Custom tasks can be implemented using the task registry.

---

## <span class="sk-h2-span">Modes</span>

Each task supports the following operational **modes**:

- **Data**: Retrieve supported datasets and generate tfrecords.
- **Train**: Train a model using a config file or inline args.
- **Evaluate**: Benchmark model performance on validation/test sets.
- **Export**: Export trained model to TFLite/TFLM formats for deployment.
- **Demo**: Run on-device inference demos using PC or Ambiq EVB.

---


## <span class="sk-h2-span">Datasets</span>

SoundKit includes a flexible dataset factory that supports both speech and non-speech corpora, as well as labeled data for supervised tasks. The following datasets are supported for SE, VAD, and KWS tasks:

### Speech Corpora
* [**LibriSpeech**](https://www.openslr.org/12) (train-clean-100, train-clean-360, dev-clean, test-clean): Large-scale read English speech corpus.
* [**THCHS-30**](https://www.openslr.org/18): Mandarin speech corpus with train/dev splits.

### Noise Datasets
* [**WHAM! Noise**](https://wham.whisper.ai/): Background noise recordings with train/val splits.
* [**MUSAN**](https://www.openslr.org/17): Contains music and noise clips suitable for data augmentation and robust training.
* [**FSD50K**](https://zenodo.org/record/4060432): Open dataset with diverse non-verbal sound events.
* [**ESC-50**](https://github.com/karoldvl/ESC-50): Environmental sound classification dataset for non-speech events.

### Reverb
* [**RIRS_NOISES**](https://www.openslr.org/28): Room impulse responses for augmenting audio with realistic reverberation.

Each dataset is loaded using a factory function and supports automatic file discovery or label mapping where applicable.


---

## <span class="sk-h2-span">Model Zoo</span>

Pre-trained models are available for SE, VAD, and KWS tasks. Each model includes:

* Downloadable `.tflite` binaries
* Training configuration files
* Evaluation reports (accuracy, F1-score, latency)
* Deployment instructions

Visit the [Model Zoo](https://ambiqai.github.io/soundkit/zoo) to explore and benchmark available models.

---

## <span class="sk-h2-span">Guides</span>

Explore the [Guides](https://ambiqai.github.io/soundkit/quickstart) section for:

* Task-specific tutorials (KWS, VAD, SE)
* Dataset preparation and augmentation
* Model customization and benchmarking
* Deployment on Ambiq Apollo EVBs

---

## <span class="sk-h2-span">License</span>

See [LICENSE](./LICENSE) for full terms.

---
