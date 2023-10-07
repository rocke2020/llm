# a0 a1_fast_tokenizer a2_metrics
# export HF_DATASETS_OFFLINE=1
file=datasets_use/a2_metrics.py
nohup python $file \
> $file.log 2>&1 &