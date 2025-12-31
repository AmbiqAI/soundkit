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