""" Basic usage example: 

from utils_comm.file_uilt import file_util

file_util.write_json({"a": 1}, "test.json")
"""

import dataclasses
import hashlib
import json
import math
import socket
from decimal import Decimal
from pathlib import Path
from typing import List, Tuple

import numpy
import pandas as pd
import yaml
from pandas import DataFrame

from utils_comm.log_util import logger


class FileUtil(object):
    """
    文件工具类
    """

    @classmethod
    def read_lines_from_txt(cls, file_path) -> List[str]:
        """
        读取原始文本数据，每行均为纯文本，自动删除头尾空格 strip()
        """
        all_raw_text_list = []
        with open(file_path, "r", encoding="utf-8") as raw_text_file:
            for item in raw_text_file:
                item = item.strip()
                all_raw_text_list.append(item)

        return all_raw_text_list

    @classmethod
    def write_lines_to_txt(cls, texts, file_path):
        """
        写入文本数据，每行均为纯文本, 自动增加换行
        """
        with open(file_path, "w", encoding="utf-8") as f:
            for item in texts:
                f.write(f"{item}\n")

    @classmethod
    def read_json(cls, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    @classmethod
    def read_jsonl(cls, file_path):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        logger.info(f"Read jsonl file {file_path} with {len(data)} lines")
        return data

    @classmethod
    def write_json(cls, data, file_path, ensure_ascii=False):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, indent=4, cls=JSONEncoder)

    @classmethod
    def write_jsonl(cls, data, file_path, ensure_ascii=False):
        with open(file_path, "w", encoding="utf-8") as f:
            for line in data:
                f.write(json.dumps(line, ensure_ascii=ensure_ascii) + "\n")

    @classmethod
    def read_yml(cls, file_path):
        """ """
        with open(file_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        config = Bunch(**config)
        return config


file_util = FileUtil()


def read_seqs_from_file(seqs_file, seq_column_name="Sequence", note=""):
    """input_file must be txt or csv, or xlsx"""
    seqs_file = Path(seqs_file)
    assert seqs_file.exists(), f"{note} input_file not exists: {seqs_file}"
    if seqs_file.suffix == ".txt":
        seqs = FileUtil.read_lines_from_txt(seqs_file)
    elif seqs_file.suffix == ".csv":
        df = pd.read_csv(seqs_file)
        assert seq_column_name, "seq_column_name is none or empty"
        seqs = get_seqs_from_df(df, seq_column_name)
    elif seqs_file.suffix == ".xlsx":
        df = pd.read_excel(seqs_file, sheet_name=0)
        assert seq_column_name, "seq_column_name is none or empty"
        seqs = get_seqs_from_df(df, seq_column_name)
    else:
        raise ValueError(
            f"{note} input_file must be txt or csv, or xlsx, but got {seqs_file.suffix}"
        )
    _seqs = []
    for seq in seqs:
        if seq:
            _seqs.append(seq)
    assert _seqs, f"{note} seqs is empty in {seqs_file}"
    logger.info("%s seqs num %s", note, len(_seqs))
    return _seqs


def get_seqs_from_df(df: DataFrame, seq_column_name="Sequence"):
    """ """
    df.dropna(subset=[seq_column_name], inplace=True)
    df.drop_duplicates(subset=[seq_column_name], inplace=True)
    seqs = df[seq_column_name].tolist()
    return seqs


def dataclass_from_dict(klass, dikt):
    try:
        fieldtypes = klass.__annotations__
        return klass(**{f: dataclass_from_dict(fieldtypes[f], dikt[f]) for f in dikt})
    except AttributeError:
        # Must to support List[dataclass]
        if isinstance(dikt, (tuple, list)):
            return [dataclass_from_dict(klass.__args__[0], f) for f in dikt]
        return dikt


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, (Tuple, set)):
            return list(o)
        if isinstance(o, bytes):
            return o.decode()
        if isinstance(o, numpy.ndarray):
            return o.tolist()
        return super().default(o)


def get_partial_files(input_files, total_parts_num=-1, part_num=-1, start_index=-1):
    """part_seq starts from 1.

    If start_index > 0, directly get partial input_files[start_index:]\n
    elseIf part_num > 0 and total_parts_num > 1, split input files\n
    else, keep orig input files
    """
    logger.info(f"Total input files num {len(input_files)}")
    if start_index > 0:
        logger.info("Get parts from index %s", start_index)
        partial_files = input_files[start_index:]
        logger.info(f"Current partial_files num {len(partial_files)}")
    elif total_parts_num > 1 and part_num > 0:
        logger.info(
            "Total_parts num %s, current part_num %s", total_parts_num, part_num
        )
        input_files_num = len(input_files)
        num_per_part = math.ceil(input_files_num / total_parts_num)
        start_i = (part_num - 1) * num_per_part
        end_i = part_num * num_per_part
        partial_files = input_files[start_i:end_i]
        logger.info(f"Current partial_files num {len(partial_files)}")
    else:
        partial_files = input_files
    return partial_files


def sort_partial_files(
    input_dir, file_suffix, total_parts=2, part_num=1, reverse=False
):
    """
    Args:
        input_dir: input dir
        file_suffix: file suffix, eg: .fasta
        total_parts: total parts num of input files
        part_num: part num, starts from 1
        reverse: reverse files order
    """
    assert file_suffix.startswith(
        "."
    ), f"file_suffix must starts with ., but got {file_suffix}"
    input_files = []
    fasta_files = sorted(Path(input_dir).glob(f"*{file_suffix}"))
    for file in fasta_files:
        input_files.append(str(file))
    if not input_files:
        raise ValueError(f"No input files found in directory: {input_dir}")
    partial_files = get_partial_files(input_files, total_parts, part_num)
    if len(partial_files) == 0:
        raise RuntimeError(
            f"No partial input files found for total_parts={total_parts}, part_num={part_num}"
        )
    if reverse:
        partial_files.reverse()
    return partial_files


def calculate_file_md5(filename):
    """For small file"""
    with open(filename, "rb") as f:
        bytes = f.read()
        readable_hash = hashlib.md5(bytes).hexdigest()
        return readable_hash


def calculate_file_md5_large_file(filename):
    """For large file to read by chunks in iteration."""
    md5_hash = hashlib.md5()
    with open(filename, "rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
        return md5_hash.hexdigest()


def calc_seq_hash(seq: str):
    """ """
    seq_sha3_name = hashlib.sha3_256(seq.encode("utf-8")).hexdigest()
    return seq_sha3_name


def get_local_ip(only_last_address=True) -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(("192.255.255.255", 1))
        local_ip = s.getsockname()[0]
    except OSError as e:
        logger.info("cannot get ip with error %s\nSo the local ip is 127.0.0.1", e)
        local_ip = "127.0.0.1"
    finally:
        s.close()
    logger.info("full local_ip %s, only_last_address %s", local_ip, only_last_address)
    if only_last_address:
        local_ip = local_ip.split(".")[-1]
    return local_ip


class Bunch(dict):
    """Container object exposing keys as attributes.

    Benefits 1: Use this class to let dir() to return the dict items.
    Benefits 2: directly use a.aa, not needing a['aa'], easierly for config access.

    Bunch objects are sometimes used as an output for functions and methods.
    They extend dictionaries by enabling values to be accessed by key,
    `bunch["value_key"]`, or by an attribute, `bunch.value_key`.

    Examples
    --------
    >>> from sklearn.utils import Bunch
    >>> b = Bunch(a=1, b=2)
    >>> b['b']
    2
    >>> b.b
    2
    >>> b.a = 3
    >>> b['a']
    3
    >>> b.c = 6
    >>> b['c']
    6
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    # def __setstate__(self, state):
    #     # Bunch pickles generated with scikit-learn 0.16.* have an non
    #     # empty __dict__. This causes a surprising behaviour when
    #     # loading these pickles scikit-learn 0.17: reading bunch.key
    #     # uses __dict__ but assigning to bunch.key use __setattr__ and
    #     # only changes bunch['key']. More details can be found at:
    #     # https://github.com/scikit-learn/scikit-learn/issues/6196.
    #     # Overriding __setstate__ to be a noop has the effect of
    #     # ignoring the pickled __dict__
    #     pass


def get_sorted_index(lst, reverse=False):
    """ """
    sorted_index = [
        i for i, x in sorted(enumerate(lst), key=lambda x: x[1], reverse=reverse)
    ]
    return sorted_index
