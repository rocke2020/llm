import json
import re, os
import logging
from multiprocessing import Pool
import nltk
from pathlib import Path
from tqdm import tqdm
from functools import partial
import unicodedata


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
SENTENCES = 'sentences'
blanks_pat = re.compile(r'\s+')
SHORTEST_LENGTH = 3


def de_accent(s):
    """ https://stackoverflow.com/a/518232/2809427 """
    return "".join([c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c)])


if __name__ == "__main__":
    from argparse import ArgumentParser


    parser = ArgumentParser()
    parser.add_argument("--pubmed_json_dir", default=None, type=str, required=True)
    args = parser.parse_args()    