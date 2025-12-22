SHELL := /bin/bash


setup:
	uv venv --python=3.10 && \
	source .venv/bin/activate && \
	uv pip install torch==2.8.0 torchaudio==2.8.0 --torch-backend=cpu && \
	git clone https://huggingface.co/smtiitm/FastSpeech2_HS_latest_models Fastspeech2_HS && \
	uv pip install -r requirements.txt && \
	cp main.py Fastspeech2_HS && cp utilities.py Fastspeech2_HS && cd Fastspeech2_HS


cleanup:
	rm -rf .venv Fastspeech2_HS