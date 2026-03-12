# Conformer

## Installating
1. Install the required dependencies:
```
uv venv --python=3.10
```
2. Install the dependencies
```
uv pip install torch==2.8.0 torchaudio==2.8.0 --torch-backend=cpu
uv pip install -r requirements.txt
```

## Benchmark
```
python bench.py \
    --model ai4bharat/indic-conformer-600m-multilingual \
    --shape 1,480000 \
    --dtype bfloat16 \
    --decoder rnnt \
    --language hi
```

NOTE - The model with CTC decoder only supports BS1 due to the backend code compiled for BS1. We are working on enabling support for larger batch sizes in the future.


## Benchmarking with Batch Size

1. To change enable higher batch size support, clone the repository in the current active dir.
```
sudo apt install git-lfs install
git clone https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
```
2. Change the config file to point the `model_onnnx_1b_batched_rnnt`. Copy paste the content below in the `config.json` file of the cloned repository.
```
{
    "auto_map": {
    "AutoConfig": "model_onnx_1b_batched_rnnt.IndicASRConfig",
    "AutoModel": "model_onnx_1b_batched_rnnt.IndicASRModel"
    },
    "BLANK_ID": 256,
    "RNNT_MAX_SYMBOLS": 10,
    "PRED_RNN_LAYERS": 2,
    "PRED_RNN_HIDDEN_DIM": 640,
    "SOS": 256
}
```

3. Run the benchmark script with the new model path and the desired batch size.
```
    python bench_batched.py \
        --model indic-conformer-600m-multilingual \
        --shape 4,480000 \
        --dtype bfloat16 \
        --decoder rnnt \
        --language hi
```


## Profiling
1. Run the script to profile the code using pytorch profiler:
    ```
    python pt_profile.py --audio samples/hindi.wav --warmup 3 --active 1 --repeat 1 --output ./profiled_output --dtype bfloat16
    ```

