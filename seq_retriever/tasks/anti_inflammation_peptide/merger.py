import os
import random
import sys

import numpy as np
from loguru import logger

sys.path.append(os.path.abspath("."))
from seq_retriever.tasks.anti_inflammation_peptide import (
    abstarct_llama3_8_8B_result_file,
    abstract_input_file,
    llama3_8_8B_result_file,
    pmc_input_file,
    table_seq_result_file,
    merged_result_file,
    manually_annotated_file,
)
from seq_retriever.utils_comm.result_merger import Merger

SEED = 0
random.seed(SEED)
np.random.seed(SEED)


def main():
    """Merge the result of the paragraph and abstract prediction

    pmc, acticle num = 1984, len(valid_df) = 1902, len(natural_df) = 1609
    abstract, acticle num = 6241, len(valid_df) = 836, len(natural_df) = 673
    table, acticle num = 1984, len(valid_df) = 1323, len(natural_df) = 1134
    """
    merger = Merger(
        pmc_result_file=llama3_8_8B_result_file,
        pmc_data_file=pmc_input_file,
        abstr_result_file=abstarct_llama3_8_8B_result_file,
        abstr_data_file=abstract_input_file,
        table_result_file=table_seq_result_file,
        pickup=0,
        overwrite=1,
        merged_result_file=merged_result_file, # type: ignore
        manually_annotated_file=manually_annotated_file, # type: ignore
        plot=0,
    )
    # merger.merge_paragraph_pred()
    # merger.merge_abstract_pred()
    # merger.merge_table_pred()
    merger.merge()


if __name__ == "__main__":
    main()