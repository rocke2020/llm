# export CUDA_VISIBLE_DEVICES=0
    # --config_overrides="n_embd=1024,n_head=16,n_layer=48,n_positions=102" \ 
# dataset_config_name wikitext-2-raw-v1 wikitext-103-raw-v1
export HF_DATASETS_OFFLINE=1
file=gpt2/run_clm.py
nohup python $file \
    --model_type gpt2 \
    --tokenizer_name /mnt/nas1/models/gpt2 \
    --cache_dir /mnt/nas1/huggingface/cache \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir tmp/test-clm-fresh \
    --overwrite_output_dir \
    --per_device_eval_batch_size 6 \
    --per_device_train_batch_size 6 \
    --num_train_epochs 30 \
    --load_best_model_at_end \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 5 \
    --bf16 \
    --dataloader_num_workers 4 \
    > $file-fresh.log 2>&1 &