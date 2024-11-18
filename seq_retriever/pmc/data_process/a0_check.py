import json
import math
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from icecream import ic
from loguru import logger
from nltk import word_tokenize
from pandas import DataFrame
from tqdm import tqdm

# nltk.download('punkt')


sys.path.append(os.path.abspath('.'))
from utils_comm.file_util import file_util
from a10_normalize import (
    data_dir,
    pmc_more_merged_file,
    section_pat,
    sections_excluded_more_file,
    sections_excluded_repo_file,
)

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

test_sections_file = 'app/tasks/pmc/analyze_data/sections_excluded.log'

def check_first_raw_sections(pmc_file):
    """ check duhan's first raw data """
    with open(pmc_file) as f:
        pmc = json.load(f)
    ic(len(pmc))
    sections = set()
    for item in pmc:
        for para in item['paragraph']:
            section = para['section']
            section = section_pat.sub('', section)
            if section:
                sections.add(section)
        ic(section)
        break
    _sections = sorted(sections)
    ic(len(_sections))
    
    with open(test_sections_file, 'w') as f:
        for section in _sections:
            f.write(f'{section}\n')


def check_para_words_num():
    """  
        para_word_nums  approximated_token_nums
    count     8884.000000              8884.000000
    mean      1054.676835              1406.235780
    std       1159.403982              1545.871976
    min          1.000000                 1.333333
    25%        173.000000               230.666667
    50%        749.000000               998.666667
    75%       1524.250000              2032.333333
    max      19643.000000             26190.666667
    """
    with open(pmc_more_merged_file, 'r', encoding='utf-8') as f:
        pmc = json.load(f)
    para_word_nums = []
    approximated_token_nums = []
    verbose = 1
    for item in pmc:
        for para in item['paragraph'].values():
            words = word_tokenize(para)
            para_word_nums.append(len(words))
            approximated_token_nums.append(len(words) / 75 * 100)
            if verbose:
                ic(para, words)
                verbose =0
    para_word_nums = np.array(para_word_nums)
    approximated_token_nums = np.array(approximated_token_nums)
    df = DataFrame({'para_word_nums': para_word_nums, 'approximated_token_nums': approximated_token_nums})
    logger.info(f'\n{df.describe()}')
    ax = df[["approximated_token_nums"]].plot.hist(bins=20).get_figure()
    ax.savefig('app/tasks/pmc/analyze_data/approximated_token_nums.png')


def check_sections():
    """  
    Input format:
        [
            {
                "pmc": "PMC7023394",
                "pmid": "31936124",
                "paragraph": {
                    "Introduction": "Peptides, polymers composed of amino acids,..."
                }
            },
        ]    

    section names start with "data":
        dataset
        data availability
        data and materials availability
        data availability statement
        data sharing statement
    """
    with open(pmc_more_merged_file) as f:
        pmc = json.load(f)
    ic(len(pmc))

    sections_to_check = set()
    section_prefix_to_check = 'data'
    for item in pmc:
        for section in item['paragraph'].keys():
            if section.lower().startswith(section_prefix_to_check):
                sections_to_check.add(section.lower())
    file_util.write_lines_to_txt(sections_to_check, test_sections_file)


if __name__ == '__main__':
    # check_first_raw_sections()
    # check_para_words_num()
    check_sections()
