SHELL := /bin/bash

setup:
	uv venv --python=3.10 && \
	source .venv/bin/activate && \
	git clone https://github.com/VarunGumma/IndicTransToolkit.git && \
	cd IndicTransToolkit && \
	uv pip install --editable ./ && \
	cd .. && \
	uv pip install -r requirements.txt && \
	uv pip install torch==2.8.0 --torch-backend=cpu