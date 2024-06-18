#!/bin/bash

sudo apt-get update
sudo apt-get install libsndfile1-dev ffmpeg unzip -y

pip install git+https://github.com/huggingface/optimum-habana.git

git clone https://github.com/gokulkarthik/Trainer 
cd Trainer && pip install -e .[all] && cd ..

git clone https://github.com/gokulkarthik/TTS 
cd TTS && pip install -e .[all] 
cd ..

echo "Done TTS, starting Indic-TTS: $(pwd)"

git clone https://github.com/AI4Bharat/Indic-TTS Indic_TTS
echo "Cloned Indic_TTs: pwd $(ls)"
pip install -r Indic_TTS/requirements.txt
pip install numpy==1.23
pip install -r Indic_TTS/inference/requirements-ml.txt 
pip install -r Indic_TTS/inference/requirements-utils.txt
pip install -r Indic_TTS/inference/requirements-server.txt

mkdir -p Indic_TTS/inference/checkpoints/
mkdir -p Indic_TTS/inference/models/v1/

wget https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/bn.zip &> /dev/null
unzip -qq bn.zip
cp -r bn Indic_TTS/inference/checkpoints/
cp -r bn Indic_TTS/inference/models/v1/

pip install ai4bharat-transliteration asteroid
pip install transformers==4.38.2
# pip install --upgrade pyworld

#! TODO: Need to find a more permanent solution
echo "CHanging transformer file!"
ls /usr/local/lib/python3.10/dist-packages
sed '122i\            return' /usr/local/lib/python3.10/dist-packages/fairseq/modules/transformer_layer.py  
sed -i '122i\            return' /usr/local/lib/python3.10/dist-packages/fairseq/modules/transformer_layer.py  