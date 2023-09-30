# 
export HF_DATASETS_OFFLINE=1
file=datasets_use/a0.py
nohup python $file \
> $file.log 2>&1 &