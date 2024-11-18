import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from icecream import ic
from loguru import logger

sys.path.append(os.path.abspath("."))
from seq_retriever.pmc.data_process import (
    orig_pmc_file,
    pmc_input_file,
    raw_table_data_file,
    table_seq_result_file,
)
from seq_retriever.pmc.table_process.save_table_data import save_table_data_file
from utils_comm.file_util import file_util

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
amino_acid_to_peptide = re.compile(re.escape("Amino Acid"), re.IGNORECASE)
wrong_name_seq = re.compile(r"^[0-9.\n>-]*$")
caption_filters = ["PCR", "primer", "gene "]
filter_strs_table_columns = ["5′", "3′", "coverage"]


def find_name_seq_index(table):
    table_columns = table["table_columns"]
    captiion = table["caption"]
    name_index = -1
    seq_index = -1
    for one_columns in table_columns:
        for index, item in enumerate(one_columns):
            if "Amino acid position" in item:  # specific process: PMC9216351
                seq_index = index
            item = re.sub(amino_acid_to_peptide, "peptide", item)
            if any(filter_str in item for filter_str in filter_strs_table_columns):
                continue

            item = item.lower()
            if (
                "peptide" in item and name_index < 0
            ):  # 第一个含有peptie的为名字，后面有name时可覆盖
                name_index = index
            elif item == "name":
                name_index = index
            if "sequence" in item and seq_index < 0:
                seq_index = index
            elif "search peptide" in item:
                seq_index = index
            if "peptide" in captiion and item.startswith("sequenc") and name_index < 0:
                name_index, seq_index = index, index
    return name_index, seq_index


def combine_peptide(table, name_index, seq_index, pmc, peptide_num):
    """
    table_content:
    [
        [
            "OIR1",
            "1",
            "Cyclo-(IRPIRP)",
            "6",
            "732.95",
            "732.97",
            "+2",
            "10.32"
        ],
        [
            "OIR2",
            "2",
            "Cyclo-(IR)2P(IR)2P",
            "10",
            "1271.62",
            "1271.66",
            "+4",
            "10.66"
        ]
    ]
    """
    result = defaultdict(set)
    name_count = defaultdict(dict)
    table_content = table["table_values"]
    for line in table_content:
        if pmc == "PMC9216351":
            peptide_name = "".join(line[name_index + 1 : len(line) - 1])
            peptide_seq = "".join(line[seq_index + 1 : len(line) - 1])
        else:
            peptide_name = line[name_index]
            peptide_seq = line[seq_index]
        peptide_name, peptide_seq = post_process(peptide_name, peptide_seq, pmc)
        for index, item in enumerate(peptide_seq):
            if item in ["", "−"]:
                continue
            item = item.rstrip(",")
            name = peptide_name[index].rstrip(",")
            name_count[name] = name_count.get(name, 0) + 1  # count the number of name
            if name_count[name] == 1:
                result[name].add(item)
                peptide_num += 1
            elif (
                name_count[name] != 1 and item != list(result[name])[0]
            ):  # preserve multiple sequences of one name
                name = f"{name}_{name_count[name]}"
                result[name].add(item)
                peptide_num += 1
    return result, peptide_num


def post_process(peptide_name, peptide_seq, pmc):
    if (
        peptide_seq
    ):  # make sure have sequence, filter the sequence which is image, like 35455455 36496664
        peptide_name, peptide_seq = replace_wrong_name_seq(peptide_name, peptide_seq)
        peptide_seq = reb_wrong_symbol(peptide_seq)
        peptide_name, peptide_seq = split_multiple_peptide(
            peptide_name, peptide_seq, pmc
        )
        peptide_seq = filter_wrong_seq(peptide_seq)
    return peptide_name, peptide_seq


def replace_wrong_name_seq(peptide_name, peptide_seq):
    """
    some peptide name(sequence) has worng type, but sequence(name) is correct.
    example:
    "ALLSISSF": "-"
    "1": "F V P W F S K F [k G R I E]",
    "2479.2": "LSQSKVLPVPQKAVPYPQRDMP",
    "1": "IWCKDDQNPHSR",
    "0.62": "ALPMHIR",
    """
    if re.fullmatch(wrong_name_seq, peptide_name):
        logger.info(f"wrong peptide name is {peptide_name = }")
        peptide_name = peptide_seq
        logger.info(
            f"peptide name is wrong, and switch {peptide_name = } to {peptide_seq = }"
        )
    if re.fullmatch(wrong_name_seq, peptide_seq):
        logger.info(f"wrong peptide sequence is {peptide_seq = }")
        peptide_seq = peptide_name
        logger.info(
            f"peptide sequence is wrong, and switch {peptide_seq = } to {peptide_name = }"
        )
    return peptide_name, peptide_seq


def reb_wrong_symbol(peptide_seq):
    """
    example:
    "RAGLQFPVGRLLRRLLR-GG-\nRRWFRRRRRR"  → "RAGLQFPVGRLLRRLLR-GG-RRWFRRRRRR"
    Stearyl- AGYLLGKLLOOLAAAALOOLL-NH2 → Stearyl-AGYLLGKLLOOLAAAALOOLL-NH2
    YGRKKRRQRRR -GLQERRGSNVSLTLDM → YGRKKRRQRRR-GLQERRGSNVSLTLDM
    """
    peptide_seq = re.sub("-\n", "-", peptide_seq)
    peptide_seq = re.sub("- ", "-", peptide_seq)
    peptide_seq = re.sub(" -", "-", peptide_seq)
    return peptide_seq


def split_multiple_peptide(peptide_name, peptide_seq, pmc):
    """
    The name may contain multiple peptides
    example：
    "Val-Pro-Pro, Ile-Pro-Pro": "Val-Pro-Pro, Ile-Pro-Pro" → [Val-Pro-Pro, Ile-Pro-Pro], [Val-Pro-Pro, Ile-Pro-Pro]
    "R7, C-R5, C-R7,\nC-r7": "RRRRRRR-NH2, C-s-s-CRRRRR-NH2, C-s-s-CRRRRRRR-NH2, C-s-s-crrrrr-NH2"
    """
    if pmc == "PMC8877061":
        # especial sequence: "F V P W F S K F[k G R I E]"
        peptide_name = re.sub(" ", "", peptide_name)
        peptide_seq = re.sub(" ", "", peptide_seq)
        return [peptide_name], [peptide_seq]

    name_list = re.split(r" |\n|\u2009", peptide_name)
    seq_list = re.split(r" |\n|\u2009", peptide_seq)
    name_len = len(name_list)
    seq_len = len(seq_list)
    if seq_len > name_len:
        """
        "R6 and CARP 6-mers": "RRRRRR-NH2, RRRRWW-NH2, rrrrrw-NH2, rrrrww-NH2,\nAc-MCRRKR-NH2, Ac-LCRRKF-NH2, Ac-RRWWIR-NH2"
        """
        name_list = seq_list
    elif seq_len < name_len:
        split_len = int(name_len / seq_len)
        name_list = [
            "".join(name_list[i * split_len : (i + 1) * split_len])
            for i in range(seq_len)
        ]
    logger.info(
        f"peptide information is splited, and results are {name_list = } and {seq_list = }"
    )
    return name_list, seq_list


def filter_wrong_seq(peptide_seq):
    seq_parttern = re.compile(r"[A-Z]{3,}|[A-Z][a-z]{2}-|-NH2")
    for index, item in enumerate(peptide_seq):
        if not re.findall(seq_parttern, item):
            peptide_seq[index] = ""
    return peptide_seq


def retrieve_pep_seq(table_data_file, out_file, skip_special_table_pmc=0):
    table_data = file_util.read_json(table_data_file)
    logger.info(f"{table_data_file = }, {len(table_data) = }")
    all_reslut = []
    pmc_num = 0
    table_num = 0
    peptide_num = 0
    for item in table_data:
        pmc_result = {}
        pmc = item["pmc"]
        pmid = item["pmid"]
        tables = item["tables"]
        pmc_infor = {"pmc": pmc, "pmid": pmid}
        pmc_result = {}
        for table in tables:
            caption = table["caption"]
            if any(filter in caption for filter in caption_filters):
                continue
            name_index, seq_index = find_name_seq_index(table)
            if name_index >= 0 and seq_index >= 0:
                logger.info(f"{pmc = }")
                result, peptide_num = combine_peptide(
                    table, name_index, seq_index, pmc, peptide_num
                )
                if result:
                    pmc_result["{}".format(table["label"])] = result
                    table_num += 1
                    logger.info(f"{pmc_result = }")
        if len(pmc_result) >= 1:
            pmc_infor["peptide_information"] = pmc_result
            pmc_num += 1
            all_reslut.append(pmc_infor)
    logger.info(f"The number of paper with peptide sequence is {pmc_num}")
    logger.info(f"The number of table with peptide sequence is {table_num}")
    logger.info(f"The number of peptide sequence is {peptide_num}")
    file_util.write_json(all_reslut, out_file)
    logger.info(f"The result saved in {out_file}")


def retrieve_pep_seq_from_table(
    pmc_with_table_file, table_data_file, table_result_file, pmc_input_file=None
):
    """run the process to retrieve the peptide sequence from the table data

    Args:
        pmc_with_table_file: the orig pmc file with the table data
        table_data_file: the file only with the table data
        table_seq_result_file: the file to save the result
        pmc_input_file: the file only with the pmc data, used to compare the pmc ids
    """
    if not Path(table_data_file).is_file():
        logger.info(f"{table_data_file} is exist, skip the process")
        save_table_data_file(pmc_with_table_file, table_data_file, pmc_input_file)
    retrieve_pep_seq(table_data_file, table_result_file)


if __name__ == "__main__":
    retrieve_pep_seq_from_table(
        orig_pmc_file, raw_table_data_file, table_seq_result_file, pmc_input_file
    )