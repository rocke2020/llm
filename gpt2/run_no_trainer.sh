export CUDA_VISIBLE_DEVICES=3
file=gpt2/run_clm_no_trainer.py
nohup python $file \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir /tmp/test-clm \
    --overwrite_output_dir \
    --per_device_eval_batch_size 6 \
    --per_device_train_batch_size 6 \
    --num_train_epochs 1 \
    > $file.log 2>&1 &