SHELL := /bin/bash


setup:
	uv venv --python=3.10 && \
	source .venv/bin/activate && \
	git clone -b New-Models https://github.com/smtiitm/Fastspeech2_HS && \
	cd Fastspeech2_HS && \
	git checkout fc0608416a0f88f7b85dab05fe7e8425f125b802 && \
	git lfs fetch --all && \
	git lfs pull 


cleanup:
	rm -rf .venv Fastspeech2_HS