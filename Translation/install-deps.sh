#!/bin/bash

git clone https://github.com/VarunGumma/IndicTransTokenizer
cd IndicTransTokenizer
pip install --editable ./
cd ..
pip install git+https://github.com/huggingface/optimum-habana.git
pip install -r requirements.txt