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
    --model openai/whisper-large-v3 \
    --dataset google/fleurs \
    --dataset_config hi_in \
    --split test \
    --batch_size 1 \
    --num_samples 100 \
    --seed 42 \
    --language hi
```

NOTE - The model currently supports only batch size of 1 due to the backend code compiled for BS1. We are working on enabling support for larger batch sizes in the future.
