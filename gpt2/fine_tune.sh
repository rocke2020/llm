#     --do_train \
export CUDA_VISIBLE_DEVICES=0
export HF_DATASETS_OFFLINE=1
file=gpt2/run_clm.py
nohup python $file \
    --model_name_or_path /mnt/nas1/models/gpt2 \
    --cache_dir /mnt/nas1/huggingface/cache \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir tmp/test-clm \
    --overwrite_output_dir \
    --per_device_eval_batch_size 6 \
    --per_device_train_batch_size 6 \
    --num_train_epochs 1 \
    > $file.log 2>&1 &