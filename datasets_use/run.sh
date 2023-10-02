# a0 a1
# export HF_DATASETS_OFFLINE=1
file=datasets_use/a0_wikitext.py
nohup python $file \
> $file.log 2>&1 &