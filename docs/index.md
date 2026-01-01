# Welcome to SoundKit

**SoundKit** is a lightweight speech processing framework optimized for embedded AI. Built on **TensorFlow Lite (TFLite)** and **TensorFlow Lite for Microcontrollers (TFLM)**, it enables always-on voice capabilities for power-constrained devices.

Whether you're prototyping on a PC or deploying to an ultra-efficient [Ambiq's family of ultra-low power SoCs](https://ambiq.com/soc/)., SoundKit provides a streamlined path from development to deployment.

For production-grade deployment, SoundKit utilizes **HeliaRT**, Ambiq's ultra-efficient edge AI runtime. Optimized specifically for the Apollo family of SoCs, HeliaRT delivers:

- Up to 3x faster inference compared to standard LiteRT (formerly TFLM) implementations.

- Custom AI kernels that leverage Apollo510’s vector acceleration hardware.

- Improved the model quantization for int16x8 support specifically designed for high-fidelity audio and speech processing.

Integrated with **NeuralSPOT**
To simplify embedded development even further, SoundKit integrates with NeuralSPOT — Ambiq’s open-source software development kit for AI acceleration and system-level optimization. NeuralSPOT provides drivers, utilities, and example projects tailored for deploying ML workloads efficiently on Ambiq hardware.


## Key Features

- **Speech Enhancement (SE)**  
  Denoising and dereverberation for clearer speech in noisy environments.

- **Keyword Spotting (KWS)**  
  Fast, low-footprint wake word detection using TFLite/TFLM-compatible models.

- **Voice Activity Detection (VAD)**  
  Lightweight, real-time detection of speech presence to save power and reduce false triggers.

- **Speaker Verification (ID)**  
  On-device speaker ID for secure, private voice authentication—no cloud needed.

## Development Flow

1. **Start on PC**  
   Prototype and validate models using TensorFlow Lite (TFLite) directly on your desktop. This enables fast, real-time verification of model accuracy, performance, and behavior before targeting embedded hardware.

2. **Deploy to [Ambiq's family of ultra-low power SoCs](https://ambiq.com/soc/).**  
   Move to **Ambiq’s Apollo5 MCU** with **TFLM** to achieve up to **10x lower power consumption** using Ambiq **Subthreshold Power Optimized Technology (SPOT)**.

3. **Optimize & Tune**  
   Use our example pipelines, quantized models, and configuration templates to accelerate development.

## Why SoundKit?

- **TFLite / TFLM Native**: Built for TensorFlow's embedded runtimes. We also provide 16x8 bit quantization scheme for minimizing accuracy loss.
- **Embedded-First**: Validated on real hardware.
- **Modular Design**: Plug-and-play components tailored for your needs.
- **Ultra-Low Power**: Ideal for wearables and battery-powered applications.

## Use Cases

- Smart earbuds with always-on keyword spotting  
- Voice-controlled wearables and health trackers  
- Embedded smart home assistants  
- Low-power environmental sound monitoring

## Get Started

- Clone the repo and follow [QuickStart](./quickstart.md) to set up.
- Run TFLite on real-time on PC.
- Flash to [Apollo5](https://ambiq.com/apollo510/) for embedded testing.

