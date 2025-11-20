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

### Default performance
    Default performance can be measured by running:
    ```
    Transcription: ['जम्मू-कश्मीर ने जो कर दिखाया है वह देश भर के लोगों के लिए भी एक मिसाल है यहां के पुलवामा से।']
    Total Time:  0.32 seconds
    Audio Length: 10.0 seconds
    Batch Size: 1
    ``` 

### Changing precision type
    ```
    self.model = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [model_path]
    )
    self.model = self.model[0]
    self.model.to(torch.float32)  ->  self.model.to(torch.bfloat16)
    ```

    Performance:
    Transcription: ['जम्मू-कश्मीर ने जो कर दिखाया है वह देश भर के लोगों के लिए भी एक मिसाल है यहां के पुलवामा से।']
    Total Time:  0.2927 seconds
    Audio Length: 10.0 seconds
    Batch Size: 1

### TcMalloc (Not suitable for lower batches)
NOTE - Not much performance increase, for 1 batch size. It is not memory bound.

Changing the memory allocator to TcMalloc can lead to performance improvements in certain scenarios. To use TcMalloc, you need to install the `google-perftools` package and set the `LD_PRELOAD` environment variable.
1. Install google-perftools:
    ```bash
    sudo apt-get install google-perftools libgoogle-perftools-dev
    ```
2. Set the `LD_PRELOAD` environment variable before running your Python script:
    ```bash
    export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
    ```


### Using Intel Extension for PyTorch (IPEX) and torch compile
1. Install Intel Extension for PyTorch:
    ```bash
    uv pip install intel_extension_for_pytorch
    ```
2. Modify your model code to use IPEX optimizations. For example,
    ```python
    import intel_extension_for_pytorch as ipex
    ...
    self.model = ipex.optimize(self.model, weights_prepack=False)
    self.model = torch.compile(self.model, backend="ipex")
    ```


### Changing precision type


    ```
        self.model = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [model_path]
        )
        self.model = self.model[0]
        self.model.to(torch.bfloat16)  ->  self.model.to(torch.float32)
    ```

