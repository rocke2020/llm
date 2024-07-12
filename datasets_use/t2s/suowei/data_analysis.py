import json
import math
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from loguru import logger
from icecream import ic

sys.path.append(os.path.abspath('.'))
from utils_comm.file_util import file_util

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

roor_dir = Path(r'D:\corpus\t2s\suowei')
train_file = 'train_suowei_0701.json'


def analyze_train_data():
    """ 
    suowei
        0701: 432 unique sql and 1994 questions 
    """
    data = file_util.read_json(roor_dir / train_file)
    outputs = set()
    ic(len(data))
    for item in data:
        outputs.add(item['output'])
    ic(len(outputs))


def main():
    analyze_train_data()
    logger.info('end')


if __name__ == '__main__':
    main()
