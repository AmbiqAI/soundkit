#!/bin/bash
make clean
make
make deploy
make view
# cd ../python/tools

# python generic_data.py \
#     --tty /dev/serial/by-id/usb-TinyUSB_TinyUSB_Device_123457-if00 
