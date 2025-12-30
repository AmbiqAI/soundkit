# 🧪 Real-Time Demo

This page explains how to run a real-time **Speech Enhancement (SE)** demo using a trained model. Demos can be executed on embedded hardware (EVB) or directly in-browser using WebUSB.

---

## 🔧 Run `demo` Mode

```bash
soundkit -t se -m demo -c configs/se/se.yaml demo.platform=pc # or evb
```

## 🧾 Demo Parameters

| Parameter | Description |
|-----------|-------------|
| `epoch_loaded` | Model checkpoint to use for inference (`best`, `latest`, or a specific integer) |
| `platform` | Target platform for demo execution. Options: `pc` (run on local machine) or `evb` (run on embedded board). |
| `tflite_dir` | Directory containing the exported `.tflite` model |
| `evb_dir` | Path to embedded board (EVB) project directory (used for firmware build/deploy) |
| `pre_gain` | Optional gain factor applied before inference (for debugging or level adjustment) |


Example:

```yaml
demo:
epoch_loaded: best
platform: pc # or evb
tflite_dir: ./soundkit/tasks/se/tflite
evb_dir: ./soundkit/tasks/se/evb
pre_gain: 1
```

---

## 💻 Deployment Modes

### 🔌 PC

- Type
    ```bash
    soundkit -t se -m demo -c configs/se/se.yaml demo.platform=pc # or evb
    ```
- A GUI will pop up. Click start to demo
### 🔌 Embedded Board (EVB)
- Type
    ```bash
    soundkit -t se -m demo -c configs/se/se.yaml demo.platform=evb # or evb
    ```
- Open your browser on [nnse-usb-dashboard](https://ambiqai.github.io/web-ble-dashboards/nnse-usb/)
- Switch raw or enhance audio via pressing Button-0 on EVB
- If you have any connection issue or no waveform showing. See the **Troubleshooting** below:
---

## 🛠️ **Troubleshooting: Connection & Waveform Issues**

If the waveform is missing or you get "Access Denied" in the browser, follow these steps to verify your hardware and permissions.

### **1. Identify the Device**

First, ensure your Linux system sees the Ambiq device. Open a terminal and run:

```bash
lsusb | grep -i "TinyUSB"

```

**What to look for:**
You should see a line similar to this:
`Bus 003 Device 030: ID cafe:4011 TinyUSB TinyUSB Device`

* **If nothing appears:** Check your USB cable (ensure it is a data cable) and try a different port.
* **If it appears:** Note the ID `cafe:4011`. This confirms the hardware is connected.

### **2. Fix "Access Denied" (udev Rules)**

Chrome cannot access the raw USB data without specific permissions. You must create a udev rule to grant access.

1. **Create the rule file:**
    ```bash
    sudo nano /etc/udev/rules.d/99-tinyusb.rules

    ```


2. **Paste the following line:**
    ```text
    SUBSYSTEM=="usb", ATTR{idVendor}=="cafe", ATTR{idProduct}=="4011", MODE="0666", GROUP="plugdev"

    ```


3. **Save and exit:** Press `Ctrl+O`, `Enter`, then `Ctrl+X`.
4. **Reload and trigger the new rules:**
    ```bash
    sudo udevadm control --reload-rules
    sudo udevadm trigger

    ```


5. **Replug your device.**

---
