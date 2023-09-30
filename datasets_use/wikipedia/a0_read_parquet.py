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
from utils.log_util import logger
from utils.file_util import FileUtil
from tqdm import tqdm


SEED = 0
random.seed(SEED)
np.random.seed(SEED)

wikipedia_dir = '/mnt/nas1/huggingface/wikipedia/20230601/en'
file = '/mnt/nas1/huggingface/wikipedia/20230601/en/train-0001-of-0083.parquet'
blanks_pat = re.compile(r'\s{2,}')


def read_file(file=file, debug=False):
    """  """
    df = pd.read_parquet(file)
    if debug:
        ic(df.shape)
        ic(df.columns)
        ic(df['text'][0], type(df['text'][0]))
        ic(df['title'][0], type(df['title'][0]))
    items = []
    for item in df['text']:
        text = blanks_pat.sub(' ', item).strip()
        if text != '':
            items.append(text)
    if debug:
        orig_file = Path(file)
        tmp_dir = Path('datasets_use/tmp_data')
        tmp_dir.mkdir(parents=True, exist_ok=True)
        text0_file = tmp_dir / f'{orig_file.stem}.log'
        FileUtil.write_raw_text(items[:2], text0_file)
    return items


if __name__ == "__main__":
    read_file(debug=1)
