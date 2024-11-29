import os
import random
import sys

import numpy as np
from icecream import ic
from loguru import logger

sys.path.append(os.path.abspath("."))
from seq_retriever.pmc.data_process.check_language import convert_all_to_eng
from seq_retriever.pmc.data_process.preprocess import convert_abstracts_to_dict
from seq_retriever.pmc.data_process.split_and_filter import split_sections
from seq_retriever.tasks.anti_inflammation_peptide import (
    abstract_input_file,
    abstracts_dict_file,
    orig_pmc_file,
    pmc_input_all_lang_file,
    pmc_input_file,
    pmc_inputs_raw_dir,
    pmc_merged_file,
    pmc_sections_filtered_file,
    sections_json_file,
)
from utils_comm.file_util import file_util

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120
SEED = 0
random.seed(SEED)
np.random.seed(SEED)


def check_pmid():
    pmid = '30560372'
    logger.info(f'{pmid = }')
    latest_pmc_data = file_util.read_json(orig_pmc_file)
    latest_abstract_data = file_util.read_json(abstract_input_file)
    for abstr in latest_abstract_data:
        if abstr['pmid'] == pmid:
            logger.info(f'{pmid = } got in abstract')
            break
    for pmc_data in latest_pmc_data:
        if pmc_data['pmid'] == pmid:
            logger.info(f'{pmid = } got in pmc')
            break


def main():
    run_slit = 0
    if run_slit:
        split_sections(
            pmc_inputs_raw_dir,
            pmc_merged_file,
            pmc_sections_filtered_file,
            pmc_input_all_lang_file,
            orig_file=orig_pmc_file,
            sections_json_file=sections_json_file,
            abstr_file=abstract_input_file,
            abstracts_dict_file=abstracts_dict_file,
            split_long=1,
        )

    ## Must manually check the non-english judge, as LLM may make mistakes to treat englighs as non-english.
    convert_all_to_eng(
        pmc_input_all_lang_file, pmc_input_file, pmc_inputs_raw_dir, re_run=0
    )


if __name__ == "__main__":
    # convert_abstracts_to_dict(abstract_input_file, abstracts_dict_file, overwrite=1)
    main()
    # check_pmid()
    logger.info('end')