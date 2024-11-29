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
from icecream import ic
from loguru import logger
from pandas import DataFrame
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.abspath("."))
from seq_retriever.pmc.data_process import (
    pmc_input_file_v10,
    pmc_test_file,
    pmc_test_split_file,
)
from seq_retriever.pmc.data_process.text_splitter import split_long_sections
from utils_comm.file_util import file_util

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
representive_pmc_ids = [
    "PMC7023394",
    "PMC7238586",
    "PMC9461663",
    "PMC10675071",
    "PMC7052017",
    "PMC8841291",
    "PMC10526274",
    "PMC8703888",
]


def select_representive_pmc():
    data = file_util.read_json(pmc_input_file_v10)
    selected_data = []
    for item in data:
        if item["pmc"] in representive_pmc_ids:
            selected_data.append(item)
    assert len(selected_data) == len(representive_pmc_ids)
    logger.info(f"{len(selected_data) = }")
    return selected_data


def main():
    selected_data = select_representive_pmc()
    file_util.write_json(selected_data, pmc_test_file)
    split_long_sections(selected_data, pmc_test_split_file)


def copy_labels():
    root_dir = Path("/mnt/nas1/patent_data/anti-inflammation_peptide")
    raw_file = root_dir / "pmc_tests_split.json"
    orig_file = root_dir / "pmc_tests_split_labels-v1.1.1.json"
    new_file = root_dir / "pmc_tests_split_labels-v1.1.2.json"
    raw_data = file_util.read_json(raw_file)
    label_orig = file_util.read_json(orig_file)
    for item, labelled_item in zip(raw_data, label_orig):
        item["labels"] = labelled_item["labels"]
    file_util.write_json(raw_data, new_file)


def check_labels():
    pass


if __name__ == "__main__":
    # main()
    copy_labels()
    # check_labels()
    logger.info("end")