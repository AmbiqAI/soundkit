# 🎧 Speech Enhancement (SE)

Speech Enhancement (SE) is the task of improving the quality and intelligibility of a speech signal that has been degraded by background noise, reverberation, or other distortions. The goal is to recover a **clean** speech signal from a **noisy** or **reverberant** input, enabling more robust communication and accurate downstream processing (e.g., automatic speech recognition, speaker identification).

---

## 🧠 Problem Formulation

The noisy speech signal $y(t)$ can be modeled as:

$$
y(t) = x(t) * h(t) + n(t)
$$

Where:

* $x(t)$: the **clean speech** signal (what we aim to recover)
* $h(t)$: the **room impulse response** (RIR), representing reverberation from the environment
* $*$: denotes convolution
* $n(t)$: the **additive noise** from background sources
* $y(t)$: the observed **noisy and reverberant** signal

The SE task is to estimate $\hat{x}(t) \approx x(t)$ from $y(t)$, effectively removing both reverberation and noise.

---

## 🗣️ Clean vs. Noisy Speech

* **Clean speech** is recorded in quiet, controlled environments with no background noise or echo.
* **Noisy speech** is corrupted by external sounds such as traffic, music, or people talking.
* **Reverberation** results from sound reflecting off surfaces in a room, smearing the speech signal over time.

These effects can be especially problematic in real-world environments like public spaces, offices, or home assistants.

---

## 🎯 SE Target

Given a corrupted signal $y(t)$, the goal of SE is to produce a high-quality estimate of the original $x(t)$. This is typically done in the time-frequency domain using spectrogram-based representations.

The figure below illustrates the difference between a clean and noisy spectrogram:

<p align="center">
  <img src="../figs/spectrogram.png" alt="Clean vs Noisy Spectrogram" width="500"/>
</p>


* The **top** spectrogram reveals the clean signal, with clearer harmonic structure and less background clutter.
* The **bottom** spectrogram shows noisy speech with smeared formants and high energy noise across frequencies.


---

## 🧰 soundKIT for SE

soundKIT provides a full pipeline for building and deploying deep learning-based SE models:

* TFRecord-based data generation
* Feature extraction (e.g., log-power spectrogram, Mel spectrogram)
* Training with numerical or perceptual loss functions
* Evaluation with standard metrics (PESQ, STOI, SI-SDR, DNSMOS)
* Export to embedded platforms using TFLite

---

By enhancing the clarity of speech signals, SE improves both human listening experiences and machine understanding in noisy or reverberant conditions.
