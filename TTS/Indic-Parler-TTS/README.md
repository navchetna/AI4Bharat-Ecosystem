# IndicParler TTS Model Optimization
This directory contains scripts and instructions to optimize the IndicParler TTS models for better performance and efficiency.


## Setup

### Method 1: Using Makefile
1. Ensure you have `uv` (Universal Virtualenv) installed. If not, install
    ```bash
    make setup
    ```
    
2. Activate the environment
    ```bash
    source .venv/bin/activate
    ```


### Method 2: Conda
1. Create a conda environment:
    ```bash
    conda env create -f environment.yml
    ```
2. Activate the conda environment:
    ```bash
    conda activate indicparler-tts
    ```


## Run the script
1. Run the main.py
    ```bash
    python main.py
    ```


## Optimization Steps
1. Change the cache manager to tcmalloc by setting the environment variable:
    ```bash
    export LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc.so.4
    ```

2. Core pinning
    Pin the process to specific CPU cores using `numactl`:
    ```bash
    numactl -C 0-31 -m 0 python main.py
    ```