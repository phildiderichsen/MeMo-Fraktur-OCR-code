#! /bin/bash

# ucloud_ocr_provision.sh
# Script for setting up a Ucloud server for parallel OCR processing.
# Philip Diderichsen, July 2021

# Install software dependencies
sudo apt-get -y update
sudo apt-get -y install libtesseract4 tesseract-ocr libtesseract-dev libpoppler73 python3.8 python3-dev libpython3.8-dev python3.8-dev python3-venv python3.8-venv

# Download Tesseract traineddata
cd /usr/share/tesseract-ocr/4.00/tessdata/
sudo wget https://github.com/tesseract-ocr/tessdata_fast/raw/master/frk.traineddata
sudo wget https://github.com/tesseract-ocr/tessdata_fast/raw/master/dan.traineddata
sudo wget https://github.com/tesseract-ocr/tessdata_best/raw/master/script/Fraktur.traineddata

# Clone git repo with code, and install Python dependencies
cd /work/Uploads/
git clone https://github.com/phildiderichsen/MeMo-Fraktur-OCR-code.git
cd MeMo-Fraktur-OCR-code/
python3.8 -m venv venv
source ./venv/bin/activate
pip3 install Image
pip install -r requirements.txt 

# Create config file for Python code
cd /work/Uploads/MeMo-Fraktur-OCR-code/
sudo cp config/config.ini.ucloud.txt config/config.ini

# Change owner of the code to ucloud in order to be able to run it without sudo
cd /work/Uploads
sudo chown -R ucloud:ucloud MeMo-Fraktur-OCR-code/
