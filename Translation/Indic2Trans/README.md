# IndicTrans Model Optimization
This directory contains scripts and instructions to optimize the IndicTrans translation models for better performance and efficiency.

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

### Method 2: Manual Setup

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


## Optimizations Steps

### Changing precision type
    ```
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.checkpoint_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16, -> torch_dtype=torch.bfloat16 
        )
    ```

Performance Boost:
- Total Input Tokens: 97 ([18, 15, 19, 21, 24])
- Total Output Tokens: 84 ([18, 15, 16, 16, 19])
- Original Time: 5.13 seconds
- Total Time: 2.77 seconds

### TcMalloc
Changing the memory allocator to TcMalloc can lead to performance improvements in certain scenarios. To use TcMalloc, you need to install the `google-perftools` package and set the `LD_PRELOAD` environment variable.
1. Install google-perftools:
    ```bash
    sudo apt-get install google-perftools libgoogle-perftools-dev
    ```
2. Set the `LD_PRELOAD` environment variable before running your Python script:
    ```bash
    export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
    ```

Performance Boost:
- Total Input Tokens: 97 ([18, 15, 19, 21, 24])
- Total Output Tokens: 84 ([18, 15, 16, 16, 19])
- Original Time: 2.77 seconds
- Total Time: 1.95 seconds


### Core Pinning
Core pinning involves binding specific processes or threads to particular CPU cores. This can help improve performance by reducing context switching and ensuring that a process consistently runs on the same core, which can enhance cache utilization.
To implement core pinning in Python, you can use the numactl command-line utility. Here's how you can do it:
1. Install numactl if it's not already installed:  
    ```bash
    sudo apt-get install numactl
    ```
2. Use numactl to run your Python script with core pinning. For example,
    ```bash
    numactl -C 48-95 -m 2,3 python main.py
    ```
    -C - Specifies the CPU cores to bind the process to. In this example, cores 48 to 95 are used. Use physical cores only.
        To monitory the CPU core usage, you can use the `htop` command. Make sure to check that the process is running on the specified cores.
    -m - Specifies the memory nodes to use. In this example, memory nodes 2 and 3 are used.
        To monitor the memory usage, you can use the `numastat` command.
            `watch -n 0.5 numastat -p $(pgrep -n python)`

Performance Boost:
- Total Input Tokens: 97 ([18, 15, 19, 21, 24])
- Total Output Tokens: 84 ([18, 15, 16, 16, 19])
- Original Time: 2.77 seconds
- Total Time: 1.16 seconds  

Scaling:
- All cores, without pinning: 1.93 seconds
- 12 Cores, 1 Memory Node: 1.56 seconds
- 19 Cores, 1 Memory Node: 1.36 seconds
- 24 Cores, 1 Memory Nodes: 1.26 seconds
- 48 Cores, 2 Memory Nodes: 1.16 seconds

NOTE - 
    Best Performance improved when using 48 cores (physical cores only) and 2 memory nodes (1.16 seconds).
    Good performance also observed when using 24 cores and 1 memory nodes (1.26 seconds).

### Intel Extension for PyTorch (IPEX)
Intel Extension for PyTorch (IPEX) is a set of optimizations and enhancements designed to improve the performance of PyTorch models on Intel hardware. It provides features such as optimized kernels, memory management, and support for Intel-specific instructions.
To use IPEX in your PyTorch code, you need to install the `intel_extension_for_pytorch` package and modify your model code to leverage IPEX optimizations.
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

Performance Boost:
- Total Input Tokens: 97 ([18, 15, 19, 21, 24])
- Total Output Tokens: 84 ([18, 15, 16, 16, 19])
- Original Time: 1.16 seconds
- Total Time: 0.86 seconds  



### Reducing Beam Search 
Reducing the beam search also reduces the decoding time for the model. 
NOTE - This may affect the quality of the translations. It is recommended to test the translations after making this change.

    ```
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            min_length=0,
            max_length=128,
            num_beams=1,  -> num_beams=5
            num_return_sequences=1,
        )
    ```

Performance Boost:
- Total Input Tokens: 97 ([18, 15, 19, 21, 24])
- Total Output Tokens: 84 ([18, 15, 16, 16, 19])
- Original Time: 1.16 seconds
- Total Time: 0.44 seconds  