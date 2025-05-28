# 📚 Supported Datasets in SoundKit

This document lists the datasets officially supported by SoundKit for speech enhancement tasks. These include speech, noise, and reverb datasets commonly used in research and real-world scenarios.

> ⚠️ **Please ensure that you have the right to use and distribute these datasets.** Always consult and respect the original dataset licenses.

---

## 🗣️ Speech Datasets

### `train-clean-100`

* 📁 Path: `wavs/LibriSpeech/train-clean-100`
* 📌 Description: Subset of the [LibriSpeech](http://www.openslr.org/12/) corpus with 100 hours of clean speech.
* 🔖 License: [LibriSpeech License (CC BY 4.0)](https://www.openslr.org/resources/12/)

### `train-clean-360`

* 📁 Path: `wavs/LibriSpeech/train-clean-360`
* 📌 Description: Larger clean speech subset with 360 hours.
* 🔖 License: Same as above.

### `dev-clean`

* 📁 Path: `wavs/LibriSpeech/dev-clean`
* 📌 Description: Development set for LibriSpeech evaluation.
* 🔖 License: Same as above.

### `test-clean`

* 📁 Path: `wavs/LibriSpeech/test-clean`
* 📌 Description: Official test set for LibriSpeech.
* 🔖 License: Same as above.

### `thchs30`

* 📁 Path: `wavs/data_thchs30/train`, `wavs/data_thchs30/dev`
* 📌 Description: A Chinese speech dataset from [THCHS-30](http://www.openslr.org/18/).
* 🔖 License: [THCHS-30 License](http://www.openslr.org/18/)

---

## 🔉 Noise Datasets

### `wham_noise`

* 📁 Path: `wavs/noise/wham_noise/tr`, `wavs/noise/wham_noise/cv`
* 📌 Description: Background noise from the WHAM! dataset used for speech separation and enhancement.
* 🔖 License: [WHAM! License (MIT)](https://github.com/wham-data/WHAM)

### `FSD50K`

* 📁 Path: `wavs/noise/FSD50K/non_speech.csv`
* 📌 Description: Non-speech audio events from [FSD50K](https://zenodo.org/record/4060432).
* 🔖 License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

### `ESC-50-master`

* 📁 Path: `wavs/noise/ESC-50-master/non_speech.csv`
* 📌 Description: Environmental sound dataset containing 50 classes.
* 🔖 License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

### `musan`

* 📁 Path: `wavs/noise/musan/music`, `wavs/noise/musan/noise`
* 📌 Description: Contains music, speech, and noise segments. Used widely in speaker recognition and enhancement tasks.
* 🔖 License: [MUSAN License (Free for research)](http://www.openslr.org/17/)

---

## 🏠 Reverb Datasets

### `RIRS_NOISES` or `rirs_noises`

* 📁 Path: `wavs/noise/RIRS_NOISES`
* 📌 Description: A large collection of simulated and real room impulse responses.
* 🔖 License: [OpenSLR License](http://www.openslr.org/28/)

---

## 📌 Notes

* All dataset loaders are registered through `@DatasetRegistry.register("dataset_name")`.
* You can inspect or modify the registration logic in `soundkit/plugins/register_datasets.py`.
* When adding new datasets, always cite and attribute the original sources as required.

---

## 🔒 Licensing Reminder

SoundKit does not distribute any of the datasets listed above. **Users are responsible** for downloading and using the datasets according to their respective licenses.

If you are publishing research or commercializing a system based on these datasets, please check licensing requirements and attribution rules for each one.
