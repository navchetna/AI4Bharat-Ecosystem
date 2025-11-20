SHELL := /bin/bash

setup:
	sudo apt-get install sox libsox-fmt-all -y && \
	export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH && \
	uv venv --python=3.10 && \
	source .venv/bin/activate && \
	wget -O hindi.pt https://asr.iitm.ac.in/SPRING_INX/models/fine_tuned/SPRING_INX_ccc_wav2vec2_Hindi.pt && \
	uv pip install torch==2.8.0 torchaudio==2.8.0 --torch-backend=cpu && \
	uv pip install -r requirements.txt

cleanup:
	sudo rm -r .venv
	rm -r hindi.pt