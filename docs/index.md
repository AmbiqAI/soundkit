#

[![](./assets/soundkit-logo-light.png#only-light)](https://ambiqai.github.io/soundkit/)
[![](./assets/soundkit-logo-dark.png#only-dark)](https://ambiqai.github.io/soundkit/)

*An AI Development Kit for real-time audio processing on Ambiq ultra-low power devices*

## Overview

Introducing **soundKIT**, an Audio AI Development Kit (ADK) that helps you train, evaluate, and deploy real-time audio models on [Ambiq's family of ultra-low power SoCs](https://ambiq.com/soc/). It ships with datasets, efficient model architectures, and reference tasks out of the box, plus optimization and deployment routines for edge inference. The kit also includes pre-trained models and task-level demos to help you get started quickly.

At its core, soundKIT is a playground: swap datasets, architectures, tasks, and training recipes via YAML, the CLI, or directly in code.

Whether you're prototyping on a PC or deploying to [Ambiq's family of ultra-low power SoCs](https://ambiq.com/soc/), soundKIT provides a consistent path from data preparation to evaluation and deployment.

For production-grade deployment, soundKIT utilizes **HeliaRT**, Ambiq's ultra-efficient edge AI runtime. Optimized specifically for the Apollo family of SoCs, HeliaRT delivers:

- Up to 3x faster inference compared to standard LiteRT (formerly TFLM) implementations.

- Custom AI kernels that leverage Apollo510’s vector acceleration hardware.

- Improved model quantization for int16x8 support designed for high-fidelity audio and speech processing.


## Key Features

- **Playground for iteration**
  Swap datasets, models, tasks, and training recipes via YAML, CLI, or Python.

- **End-to-end workflow**
  Data prep, feature extraction, training, evaluation, and export in one place.

- **Embedded-first optimization**
  Quantization-aware workflows, streaming inference patterns, and memory-conscious models.

- **Hardware acceleration ready**
  HeliaRT kernels and NeuralSPOT integration for Apollo-class MCUs.

- **Open and extensible**
  Add new datasets, tasks, and architectures without rewriting the pipeline.

## Supported Tasks

- **Speech Enhancement (SE)**
  Denoising and dereverberation for clearer speech in noisy environments.

- **Keyword Spotting (KWS)**
  Fast, low-footprint wake word detection using TFLite/TFLM-compatible models.

- **Voice Activity Detection (VAD)**
  Lightweight, real-time detection of speech presence to save power and reduce false triggers.

- **Speaker Verification (ID)**
  On-device speaker ID for secure, private voice authentication—no cloud needed.

## Development Flow

1. **Prototype on PC**
   Iterate quickly with real-time input and consistent configs.

2. **Train and evaluate**
   Run data prep, training, and benchmarking from the same CLI workflow.

3. **Export and demo**
   Generate deployable artifacts and validate on PC or EVB with matching settings.

## Why soundKIT?

- **Made for iteration**: Mix datasets, tasks, and architectures without rewriting the pipeline.
- **Embedded-First**: Validated on real hardware.
- **Modular Design**: Plug-and-play components for data, models, and tasks.
- **Ultra-Low Power**: Ideal for wearables and battery-powered applications.

## Use Cases

- Smart home, wearables, and smart glasses
- Industrial and facilities monitoring
- Automotive in-cabin voice control and noise suppression
- Healthcare and wellness devices
- Security and public safety screening
- Consumer audio accessories and conferencing

## Modes

soundKIT provides a consistent CLI with task-level modes:

- **data**: dataset setup and preparation
- **train**: model training workflows
- **evaluate**: benchmarking, metrics, and validation
- **export**: deployment artifact generation
- **demo**: real-time inference on PC or EVB

## Datasets

soundKIT includes a flexible dataset factory that supports common speech, noise, and reverb corpora (for example LibriSpeech, THCHS-30, MUSAN, FSD50K, ESC-50, RIRS_NOISES). Datasets are not redistributed; users download and use them under their respective licenses.

## Get Started

- Clone the repo and follow [QuickStart](./quickstart.md) to set up.
- Run real-time inference on PC.
- Validate on [Apollo5](https://ambiq.com/apollo510/) with the same configs.

## Web Dashboards
To visualize data from your Apollo5 EVB via WebUSB, use the dashboards:

* [Launch ID Dashboard](web-dashboards/id-usb/index.html)
* [Launch NNSE Dashboard](web-dashboards/nnse-usb/index.html)
* [Launch SD Dashboard](web-dashboards/sd-usb/index.html)
