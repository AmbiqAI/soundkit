# 🗣️ Voice Activity Detection (VAD)

Voice Activity Detection (VAD) is the task of identifying whether a segment of audio contains human speech or not. It is a crucial pre-processing step in many real-time audio applications including speech recognition, telecommunication systems, and smart devices.

---

## 🧠 Problem Formulation

The input signal $y(t)$ is a mixture of:

- **Speech segments** that we want to detect
- **Background noise** (e.g., traffic, music, keyboard typing, etc.)
- **Silence** or non-speech audio

The goal of VAD is to estimate a binary mask $m(t)$ such that:

$$
m(t) =
\begin{cases}
1, & \text{if speech is present at time } t \\
0, & \text{otherwise}
\end{cases}
$$

---

## 🔍 Why VAD Matters

VAD enables:

- Efficient transmission in voice communication systems (e.g., VoIP, cellular)
- Noise-aware processing for speech enhancement
- Accurate speech-to-text transcription by filtering out silence
- Trigger mechanisms in voice-controlled devices (e.g., smart assistants)

---

## 📊 Clean vs. Noisy Audio

In real-world conditions, VAD must operate under:

- **Clean speech**: well-recorded and noise-free audio
- **Noisy speech**: corrupted with background interference
- **Reverberant environments**: echo and reflections

Robust VAD models detect voice reliably even in challenging conditions.

---

## 🎯 VAD Target

Given a time-domain signal $y(t)$, the VAD model computes a probability or binary decision over frames (e.g., every 10 ms) indicating speech presence.

The output can be visualized as a mask overlay on the spectrogram or a framewise binary vector.

---

## 🧰 soundKIT for VAD

soundKIT provides an end-to-end VAD pipeline that includes:

- TFRecord-based synthetic data creation with speech + noise + reverb
- Frame-level feature extraction (e.g., log-power spectrogram)
- Deep learning model training (e.g., CRNN)
- Real-time evaluation and visualization
- Export to embedded systems (TFLite, C header)

---

Robust and low-latency VAD is key to voice-first interfaces, on-device AI, and power-efficient audio systems.
