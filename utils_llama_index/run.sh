# 
gpu=$1
if [ -z $gpu ]; then
    gpu=0
fi
export CUDA_VISIBLE_DEVICES=$gpu
port=8001
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# api_client api_server llm_loader vllm_loader api_server_vllm
file=utils_llama_index/api_server_vllm.py
nohup python $file \
    --port $port \
    > $file-$port.log 2>&1 &