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

sys.path.append(os.path.abspath("."))
from utils_comm.file_util import file_util
from utils_llama_index.api_client import query_with_context

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
data_dir = Path("/mnt/nas1/patent_data/anti-inflammation_peptide")
pmc_more_merged_file = data_dir / "parsed_pmc_merged_further.json"
pmc_after_filter_file = data_dir / "pmc_after_filter.json"
sections_excluded_more_file = data_dir / "sections_excluded_further.txt"


def save_filterd_pmc():
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
    """
    data = file_util.read_json(pmc_more_merged_file)
    logger.info(f"len(data): {len(data)}")
    sections_excluded = file_util.read_lines_from_txt(sections_excluded_more_file)
    data_filtered = []
    for item in data:
        for k, v in item.items():
            if k == "paragraph":
                new_v = {}
                for sec, para in v.items():
                    if sec.lower() not in sections_excluded:
                        new_v[sec] = para
                item[k] = new_v
        data_filtered.append(item)
    file_util.write_json(data_filtered, pmc_after_filter_file)


basic_examples_question = """
Peptide sequences are defined as sequences of amino acids and mostly have two types: one uppercase letter (A-Z) for one amino acid, for examples: RKKRRQRRR, ; and three letters (Ala, Cys, Asp, Glu, Phe, Gly, His, Ile, Leu, Met, Asn, Pro, Lys, Gln, Arg, Ser, Thr, Val, Trp, Tyr) for one amino acid and concatenated by hyphen "-", for examples: Gln-Cys-Gln-Gln-Ala-Val-Gln-Ser-Ala-Val. Peptide sequences may have extra modification or molecular letters at the start or end of sequeences, for examples: H-Ile-Pro-Arg-Cys-Arg-Lys-Met-Pro-Gly-Val-Lys-Met-Cys-NH2, Pam3Cys-Ala-Gly, Cu-GHK. Peptide sequences have the amino acids number between 3 and 50, and mostly are short sequences.
Identify peptide sequences only from the context information and not prior knowledge. If the sequence is not available in the context, do not generate a sequence, but treat as unavailable. Output each peptide data with both its name and sequence on a separate line.
"""
more_examples_question = """
Peptide sequences are defined as sequences of amino acids and mostly have two types: one uppercase letter (A-Z) for one amino acid and without any , for examples: YKKHRQRCW, INLKALAALAKKIL, etc; and three letters (Ala, Cys, Asp, Glu, Phe, Gly, His, Ile, Leu, Met, Asn, Pro, Lys, Gln, Arg, Ser, Thr, Val, Trp, Tyr) for one amino acid and concatenated by hyphen "-", for examples: Gln-Cys-Gln-Gln-Ala-Val-Gln-Ser-Ala-Val. Peptide sequences may have extra modification or molecular letters at the start or end of sequeences, for examples: H-Ile-Pro-Arg-Cys-Arg-Lys-Met-Pro-Gly-Val-Lys-Met-Cys-NH2, Pam3Cys-Ala-Gly, and Cu-GHK, etc. Peptide sequences have the amino acids number between 3 and 50, and mostly are short sequences.
Find peptide sequences only from the context information and not prior knowledge. If the sequence is not available in the context above, do not use prior knowledge to generate a sequence, but directly treat as unavailable. Output each peptide data with both its name and sequence on a separate line.
"""
no_examples_question = """
Peptide sequences are defined as sequences of amino acids and mostly have two types: one uppercase letter (A-Z) for one amino acid; and three letters (Ala, Cys, Asp, Glu, Phe, Gly, His, Ile, Leu, Met, Asn, Pro, Lys, Gln, Arg, Ser, Thr, Val, Trp, Tyr) for one amino acid and concatenated by hyphen "-". Peptide sequences may have extra modification or molecular letters at the start or end of sequeences. Peptide sequences have the amino acids number between 3 and 50, and mostly are short sequences.
Identify peptide sequences only from the context information and not prior knowledge. If the sequence is not available in the context, do not generate a sequence, but treat as unavailable. Output each peptide data with both its name and sequence on a separate line.
"""
concise_question = """
Peptide sequences are defined as sequences of amino acids with the amino acids number between 3 and 50, and mostly are short sequences.
Identify peptide sequences only from the context information and not prior knowledge. If the sequence is not available in the context, do not generate a sequence, but treat as unavailable. Output each peptide data with both its name and sequence on a separate line.
"""

question = concise_question


def retrieve_seqs(test_num=1, reverse_input_data_order=0):
    """
    Args:
        test_num: int, default 1, the number of samples to test the function.
        reverse_input_data_order: int, default 0, if 1, reverse the input data order.
    """
    data = file_util.read_json(pmc_after_filter_file)
    if reverse_input_data_order:
        data = data[::-1]
        out_result_file = data_dir / "pmc_retrieved_seqs_reversed_order.json"
    else:
        out_result_file = data_dir / "pmc_retrieved_seqs.json"

    logger.info(f"{reverse_input_data_order = }, question\n{question}")
    count = 0
    if out_result_file.exists():
        all_results = file_util.read_json(out_result_file)
        pre_retrieved_pmc_ids = [item["pmc"] for item in all_results]
    else:
        pre_retrieved_pmc_ids = []
        all_results = []
    for item in tqdm(data):
        pmc = item["pmc"]
        pmid = item["pmid"]
        count += 1
        logger.info(f"{pmc = }, {count = }")
        if pmc in pre_retrieved_pmc_ids:
            logger.info(f"{pmc = } pre-retrieved, and skip")
            continue
        paragraph = item["paragraph"]
        result = {"pmc": pmc, "pmid": pmid, "paragraph": {}}
        for sec, context in paragraph.items():
            result["paragraph"][sec] = context
            llm_reply = query_with_context(question, context)
            result["paragraph"][f"{sec}_llm_reply"] = llm_reply
        all_results.append(result)

        if test_num > 0:
            if count >= test_num:
                break
        file_util.write_json(all_results, out_result_file)


if __name__ == "__main__":
    retrieve_seqs(test_num=0, reverse_input_data_order=1)
    # save_filterd_pmc()
    logger.info("end")
