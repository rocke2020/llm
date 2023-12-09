from pathlib import Path
import pandas as pd
import numpy as np
from pandas import DataFrame
import os, sys, shutil, logging, json
import re, random, math
from icecream import ic
ic.configureOutput(includeContext=True, argToStringFunction=lambda _: str(_))
ic.lineWrapWidth = 120
sys.path.append(os.path.abspath('.'))
from collections import defaultdict
from utils_comm.log_util import logger
from utils_comm.file_util import FileUtil
from tqdm import tqdm


SEED = 0
random.seed(SEED)
np.random.seed(SEED)

wikipedia_dir = '/mnt/nas1/dong-qichang/corpus/wiki/wikipedia/20230601/en/'
file = wikipedia_dir +  'train-0001-of-0083.parquet'
blanks_pat = re.compile(r'\s{2,}')


def read_file(file=file, debug=False):
    """  
    df.shape: (80253, 4)
    df.columns: Index(['id', 'url', 'title', 'text']
    df['title'][0]: Anarchism
    """
    df = pd.read_parquet(file)
    if debug:
        ic(df.shape)
        ic(df.columns)
        ic(type(df['text'][0]), type(df['title'][0]))
        ic(df['text'][0])
        ic(df['title'][0])
    items = []
    for item in df['text']:
        text = blanks_pat.sub(' ', item).strip()
        if text != '':
            items.append(text)
    if debug:
        orig_file = Path(file)
        tmp_dir = Path('datasets_use/tmp_data')
        tmp_dir.mkdir(parents=True, exist_ok=True)
        demo_texts = items[:2] + items[-2:]
        text0_file = tmp_dir / f'{orig_file.stem}.txt'
        FileUtil.write_lines_to_txt(demo_texts, text0_file)
    return items


if __name__ == "__main__":
    read_file(debug=1) # type: ignore
