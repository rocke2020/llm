import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from loguru import logger

sys.path.append(os.path.abspath("."))

from seq_retriever.pmc.data_process import abstract_file, abstracts_dict_file
from utils_comm.file_util import file_util


def convert_abstracts_to_dict(abstr_file, abstr_dict_file, overwrite=1):
    if not overwrite and abstr_dict_file.exists():
        abstracts_dict = file_util.read_json(abstr_dict_file)
        logger.info(f'{len(abstracts_dict) = }')
        return abstracts_dict
    abstracts = file_util.read_json(abstr_file)
    logger.info(f'{len(abstracts) = }')
    abstracts_dict = {}
    for abstract in abstracts:
        if abstract["pmid"] in abstracts_dict:
            logger.warning(f'{abstract["pmid"] = }')
        abstracts_dict[abstract["pmid"]] = abstract
    file_util.write_json(abstracts_dict, abstr_dict_file)
    logger.info(f'{len(abstracts_dict) = }')
    return abstracts_dict


if __name__ == "__main__":
    convert_abstracts_to_dict(abstract_file, abstracts_dict_file)