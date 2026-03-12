git clone https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
mkdir openvino_irs

# Convert the ONNX model to OpenVINO IR format
ovc --compress_to_fp16 False indic-conformer-600m-multilingual/assets/encoder.onnx
ovc --compress_to_fp16 False indic-conformer-600m-multilingual/assets/joint_enc.onnx
ovc --compress_to_fp16 False indic-conformer-600m-multilingual/assets/rnnt_decoder.onnx
ovc --compress_to_fp16 False indic-conformer-600m-multilingual/assets/ctc_decoder.onnx
ovc --compress_to_fp16 False indic-conformer-600m-multilingual/assets/joint_pred.onnx
ovc --compress_to_fp16 False indic-conformer-600m-multilingual/assets/joint_pre_net.onnx
ovc --compress_to_fp16 False indic-conformer-600m-multilingual/assets/joint_post_net_hi.onnx

mv *.xml openvino_irs/
mv *.bin openvino_irs/