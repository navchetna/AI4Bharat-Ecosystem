# FastSpeech2_HS Installation


## Setup

Option 1 - Using conda
    1. Create a new conda environment:
       ```bash
       conda create -n fastspeech2_hs python=3.10 -y
       ```
    2. Activate the environment:
       ```bash
         conda activate fastspeech2_hs
       ```
    3. Install the dependencies:
       ```bash
       pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
       pip install -r requirements.txt
       ```
    4. Clone the repository:
       ```bash
       git clone -b New-Models https://github.com/smtiitm/Fastspeech2_HS
    ```

    5. Copy all the files from the cloned repository to the current directory:
       ```bash
       cp -r *.py Fastspeech2_HS/
       ```

    6. Download the pretrained models
        ```bash
        cd Fastspeech2_HS && \
        git checkout && \ fc0608416a0f88f7b85dab05fe7e8425f125b802 && \
	git lfs fetch --all && \
    git lfs checkout
        ```

Option 2 - Using Makefile

    1. Run the make command to set up the environment and clone the repository:
       ```bash
       make setup
       ```


Option 3 - Using uv

    1. Install uv:
       ```bash
       pip install uv
       ```
    2. Create the environment and activate
       ```bash
        uv venv --python=3.10 
        source .venv/bin/activate     
       ```
    3. Install the dependencies:
       ```bash
       uv pip install torch==2.8.0 torchaudio==2.8.0 --torch-backend=cpu
       pip install -r requirements.txt
       ```

    4. Clone the repository:
       ```bash
       git clone -b New-Models https://github.com/smtiitm/Fastspeech2_HS
       ```
    5. Copy all the files from the cloned repository to the current directory:
       ```bash
       cp -r *.py Fastspeech2_HS/
       ```

    6. Download the pretrained models
        ```bash
        cd Fastspeech2_HS && git checkout fc0608416a0f88f7b85dab05fe7e8425f125b802 && \
    git lfs fetch --all && \
    git lfs checkout
        ```

**NOTE** - Due to the limit on the free tier of Git LFS, the model files may not be downloaded completely. 


## Run 
To run the FastSpeech2_HS model for inference, use the following command:

1. Enter the cloned directory:
   ```bash
   cd Fastspeech2_HS
   ```
2. Run the inference script with the desired text input:
   ```bash
   python main.py
   ```


