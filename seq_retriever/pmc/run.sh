# 
file=app/tasks/pmc/run/openchat.py
nohup python $file \
    > $file.log.2 2>&1 &