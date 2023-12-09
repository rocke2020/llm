from pathlib import Path
import pandas as pd
import numpy as np
from pandas import DataFrame
import os, sys, shutil, logging, json
import re, random, math
from icecream import ic

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120
sys.path.append(os.path.abspath("."))
from collections import defaultdict
from utils_comm.log_util import logger
from utils_comm.file_util import FileUtil
from tqdm import tqdm


SEED = 0
random.seed(SEED)
np.random.seed(SEED)

wikipedia_dir = "/mnt/nas1/dong-qichang/corpus/wiki/wikipedia/20230601/en/"
test_file = wikipedia_dir + "train-0001-of-0083.parquet"
blanks_pat = re.compile(r"\s{2,}")
en_wikipedia_jsonl_file = Path(wikipedia_dir).parent / "en_wikipedia.jsonl"


def check_file(file=test_file, debug=False):
    """
    df.shape: (80253, 4)
    df.columns: Index(['id', 'url', 'title', 'text']
    df['title'][0]: Anarchism
    """
    df = pd.read_parquet(file)
    if debug:
        ic(df.shape)
        ic(df.columns)
        ic(type(df["text"][0]), type(df["title"][0]))
        ic(df["text"][0])
        ic(df["title"][0])
    items = []
    for item in df["text"]:
        text = blanks_pat.sub(" ", item).strip()
        if text != "":
            items.append(text)
    if debug:
        orig_file = Path(file)
        tmp_dir = Path("datasets_use/tmp_data")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        demo_texts = items[:2] + items[-2:]
        text0_file = tmp_dir / f"wikipedia_{orig_file.stem}.txt"
        FileUtil.write_lines_to_txt(demo_texts, text0_file)
    return items


def convert_to_jsonl():
    sentences = []
    for file in tqdm(Path(wikipedia_dir).glob("*.parquet")):
        df = pd.read_parquet(file)
        for item in df["text"]:
            text = blanks_pat.sub(" ", item).strip()
            if len(text) > 10 and len(text.split()) > 1:
                sentences.append({"text": text})
    logger.info(f"sentences: {len(sentences)}, starts to write")
    with open(en_wikipedia_jsonl_file, "w", encoding="utf-8") as f:
        for line in sentences:
            f.write(json.dumps(line, ensure_ascii=False, indent=4) + "\n")
    logger.info('end')

if __name__ == "__main__":
    # check_file(debug=1)  # type: ignore
    convert_to_jsonl()
