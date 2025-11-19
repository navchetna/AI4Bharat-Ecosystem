# IndicWav2vec Model Optimization
This directory contains scripts and instructions to optimize the IndicWav2vec ASR models for better performance and efficiency.


## Setup

### Method 1: Using Makefile
1. Ensure you have `uv` (Universal Virtualenv) installed. If not, install it using:
    ```bash
    pip install uv
    ```
2. Run the setup command:
    ```bash
    make setup
    ``` 

### Method 2: Manual setup
1. Create a virtual environment and activate it:
    ```bash
    uv venv --python=3.11
    source .venv/bin/activate
    ```
2. Install the dependencies:
    ```bash
    uv pip install -r requirements.txt
    uv pip install torch==2.8.0 --torch-backend=cpu
    ```
3. Install the viztracer to profile the code:
    ```bash
    uv pip install viztracer    
    ``` 


## Optimization Steps
### Changing precision type
    ```
        self.model = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [model_path]
        )
        self.model = self.model[0]
        self.model.to(torch.bfloat16)  ->  self.model.to(torch.float32)
    ```