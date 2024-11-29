# 
gpu=$1
if [ -z $gpu ]; then
    gpu=0
fi
export CUDA_VISIBLE_DEVICES=$gpu
export LLAMA_INDEX_CACHE_DIR=/mnt/nas1/models/llama_index_cache

# starter_local_llm local_llm_Calme  mistral_
# query_a0 Starling_LM query_a1_use_local_api
file=/home/qcdong/codes/llamaIndex/llm_test/query_a1_use_local_api.py

# hybrid_retriever_a0_0_check_data hybrid_retriever_a0_1_Starling_LM
# file=app/tasks/patents/hybrid_retriever_a0_1_Starling_LM.py

python $file \
    2>&1  </dev/null | tee $file.log
