#!/bin/sh
apt update && apt install -y openslide-tools
pip3 install openslide-python
pip3 install tensorflow