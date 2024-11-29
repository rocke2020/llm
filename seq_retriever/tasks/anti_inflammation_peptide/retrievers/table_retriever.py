import os
import random
import sys

from loguru import logger

sys.path.append(os.path.abspath("."))
from seq_retriever.pmc.table_process.select_pep_seq_from_tables import (
    retrieve_pep_seq_from_table,
)
from seq_retriever.pmc.table_process.save_table_data import save_table_data_file
from seq_retriever.tasks.anti_inflammation_peptide import (
    orig_pmc_file,
    pmc_input_file,
    raw_table_data_file,
    table_seq_result_file,
)

SEED = 0
random.seed(SEED)


def main(overwrite_raw_table_data_file: bool = False):
    if overwrite_raw_table_data_file:
        save_table_data_file(orig_pmc_file, raw_table_data_file, pmc_input_file)
    retrieve_pep_seq_from_table(
        orig_pmc_file, raw_table_data_file, table_seq_result_file, pmc_input_file
    )


if __name__ == "__main__":
    main(overwrite_raw_table_data_file=0) # type: ignore
    logger.info("end")