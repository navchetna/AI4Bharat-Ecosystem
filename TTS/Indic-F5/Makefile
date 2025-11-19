SHELL := /bin/bash

setup:
	uv venv --python=3.10 && \
	source .venv/bin/activate && \
	uv pip install git+https://github.com/ai4bharat/IndicF5.git --torch-backend=cpu && \
	uv pip install -r requirements.txt
