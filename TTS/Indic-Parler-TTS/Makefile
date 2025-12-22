SHELL := /bin/bash


setup:
	uv venv --python=3.10 && \
	source .venv/bin/activate && git lfs install --skip-smudge && \
	uv pip install git+https://github.com/huggingface/parler-tts.git && git lfs install --force 

