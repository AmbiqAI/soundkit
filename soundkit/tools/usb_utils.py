import serial.tools.list_ports
def find_tinyusb_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if 'TinyUSB' in port.description or 'TinyUSB' in port.device:
            return port.device
    return None