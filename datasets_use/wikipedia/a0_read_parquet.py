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
from utils_comm.file_util import file_util
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
        file_util.write_lines_to_txt(demo_texts, text0_file)
    return items


def convert_to_jsonl(debug=False):
    """sentences: 6660595"""
    sentences = []
    out_file = en_wikipedia_jsonl_file
    if debug:
        out_file = Path("datasets_use/tmp_data") / "wikipedia.jsonl"
    for file in tqdm(Path(wikipedia_dir).glob("*.parquet")):
        df = pd.read_parquet(file)
        for raw_text in df["text"]:
            sub_sentences = []
            _sentences = raw_text.split("\n")
            for sent in _sentences:
                _sent = blanks_pat.sub(" ", sent).strip()
                # Each sentence should have at least 5 characters and 2 words, otherwise filter.
                if len(_sent) > 5 and len(_sent.split()) > 1:
                    sub_sentences.append(_sent)
            text = "\n".join(sub_sentences)
            if len(text) > 5:
                sentences.append({"text": text})
            if len(text) < 10:
                ic(len(text))
                ic(raw_text)
        if debug:
            break
    logger.info(f"sentences: {len(sentences)}, starts to write")

    with open(out_file, "w", encoding="utf-8") as f:
        for line in sentences:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            if debug:
                break

    logger.info("end")


if __name__ == "__main__":
    # check_file(debug=1)  # type: ignore
    convert_to_jsonl(debug=False)
