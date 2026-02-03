
# 🗣️ Keyword Spotting (KWS)

Keyword Spotting (KWS) is the task of identifying whether a specific keyword or phrase (e.g., "galaxy") occurs in an audio stream. It is a foundational technology behind wake-word engines in smart devices and low-power voice interfaces.

---

## 🧠 Problem Formulation

The input signal $y(t)$ is a continuous audio stream that may include:

- **Target keywords** to detect
- **Background speech** not containing the keyword
- **Environmental noise** and **silence**

The goal of KWS is to predict a binary or probabilistic mask $m_k(t)$ for each keyword $k$:

$$
m_k(t) =
\begin{cases}
1, & \text{if keyword } k \text{ is present at time } t \\
0, & \text{otherwise}
\end{cases}
$$

---

## 🔍 Why KWS Matters

KWS enables:

- Always-on voice interfaces in smart speakers, earbuds, and wearables
- Privacy-preserving local voice triggers without cloud processing
- Ultra-low-power wake word detection for battery-constrained applications
- Real-time command recognition in embedded systems

---

## 🎧 Noisy & Long-Tail Detection

KWS must perform reliably under:

- **Speech clutter**: background conversations and overlapping talkers
- **Noise corruption**: audio with non-speech sound sources
- **Far-field speech**: distant or reverberant keyword utterances
- **Rare-event scenarios**: few positive keyword examples in long audio

---

## 🎯 KWS Target

Given $y(t)$, the system performs framewise detection of keywords within a window (e.g., every 10 ms). The output is:

- A per-frame binary mask (1 = keyword present)
- Or a score/probability curve that peaks around keyword regions

This is useful for visualization, alignment, and event-based inference.

---

## 🧰 soundKIT for KWS

soundKIT offers a complete KWS pipeline:

- Synthetic dataset generation with keyword injection in noise
- Time-frequency feature extraction (e.g., log-power spectrogram)
- CRNN-based model training with focal loss
- Evaluation and framewise prediction visualization
- Model export to TFLite and embedded C headers for EVB use

---

Real-time keyword spotting unlocks natural and responsive voice UIs, even on ultra-low-power edge platforms.
