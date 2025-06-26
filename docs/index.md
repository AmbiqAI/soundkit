# Welcome to SoundKit

**SoundKit** is a lightweight speech processing framework optimized for embedded AI. Built on **TensorFlow Lite (TFLite)** and **TensorFlow Lite for Microcontrollers (TFLM)**, it enables always-on voice capabilities for power-constrained devices.

Whether you're prototyping on a PC or deploying to an ultra-efficient [Ambiq's family of ultra-low power SoCs](https://ambiq.com/soc/)., SoundKit provides a streamlined path from development to deployment.

## Key Features

- **Speech Enhancement**  
  Denoising and dereverberation for clearer speech in noisy environments.

- **Keyword Spotting**  
  Fast, low-footprint wake word detection using TFLite/TFLM-compatible models.

- **Voice Activity Detection (VAD)**  
  Lightweight, real-time detection of speech presence to save power and reduce false triggers.

- **Speaker Verification**  
  On-device speaker ID for secure, private voice authentication—no cloud needed.

## Development Flow

1. **Start on PC**  
   Prototype and validate models using **TFLite** on your desktop.

2. **Deploy to [Ambiq's family of ultra-low power SoCs](https://ambiq.com/soc/).**  
   Move to **Ambiq’s Apollo5 MCU** with **TFLM** to achieve up to **10x lower power consumption** using Ambiq **Subthreshold Power Optimized Technology (SPOT)**.

3. **Optimize & Tune**  
   Use our example pipelines, quantized models, and configuration templates to accelerate development.

## Why SoundKit?

- **TFLite / TFLM Native**: Built for TensorFlow's embedded runtimes.
- **Embedded-First**: Validated on real hardware.
- **Modular Design**: Plug-and-play components tailored for your needs.
- **Ultra-Low Power**: Ideal for wearables and battery-powered applications.

## Use Cases

- Smart earbuds with always-on keyword spotting  
- Voice-controlled wearables and health trackers  
- Embedded smart home assistants  
- Low-power environmental sound monitoring

## Get Started

- Clone the repo and follow `README.md` to set up.
- Run TFLite examples on PC.
- Flash to Apollo5 for embedded testing.

