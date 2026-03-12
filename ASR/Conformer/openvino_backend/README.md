# OpenVINO backend execution



## Environment Setup
1. Create environemnt and install the dependencies
    ```uv venv --python=3.10
    source .venv/bin/activate
    uv pip install torch==2.5.0 torchaudio==2.5.0 openvino onnxruntime-openvino onnx --torch-backend=cpu
    ```

2. Run the script to convert the model to OpenVINO IR format
    ```
    bash onnx_to_ovir.sh
    ```

## Convert the onnx to openvino_irs

```bash
ovc --compress_to_fp16 False <MODEL>
```

## Benchmarking with Openvino


### Encoder
 ```
 taskset -c 0-4 benchmark_app -d GPU -nstreams 1 -nthreads 1 -api sync -hint none -infer_precision f16 -m openvino_irs/encoder.xml -shape audio_signal[1,80,3001],length[1] -niter 100 
 ```

 ### Joint Encoder Decoder
 ```
taskset -c 0-4 benchmark_app -d GPU -nstreams 1 -nthreads 1 -api sync -hint none -infer_precision f16 -m openvino_irs/joint_enc.xml -shape [1,376,1024] -niter 100 
```

### Decoder

#### RNNT is static
```
taskset -c 0-4 benchmark_app -d GPU -nstreams 1 -nthreads 1 -api sync -hint none -infer_precision f16 -m openvino_irs/rnnt_decoder.xml -shape targets[1,1],target_length[1],states.1[2,1,640],onnx::Slice_3[2,1,640] -niter 100 
```

#### Joint Pred
```
taskset -c 0-4 benchmark_app -d GPU -nstreams 1 -nthreads 1 -api sync -hint none -infer_precision f16 -m openvino_irs/joint_pred.xml -shape input[1,1,640]  -niter 100
```


#### Joint Pre Net
```
taskset -c 0-4 benchmark_app -d GPU -nstreams 1 -nthreads 1 -api sync -hint none -infer_precision f16 -m openvino_irs/joint_pre_net.xml -shape input[1,1,640]  -niter 100
```


#### Joint Post Net
```
taskset -c 0-4 benchmark_app -d GPU -nstreams 1 -nthreads 1 -api sync -hint none -infer_precision f16 -m openvino_irs/joint_post_net_hi.xml -shape input[1,1,640]  -niter 100
```