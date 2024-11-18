import os
import random
import sys

import numpy as np
from loguru import logger

sys.path.append(os.path.abspath("."))
from seq_retriever.llms.llama3.run import LLMRetriever
from seq_retriever.llms.questions_prompts import (
    FULL_EXAMPLES_QUESTION,
    abstract_question,
)
from seq_retriever.tasks.anti_inflammation_peptide import (
    abstarct_result_file_base,
    abstract_input_file,
    abstract_test_file,
    llama3_result_file_base,
    pmc_input_file,
    pmc_test_data_results_dir,
    pmc_test_split_file,
    pmc_test_split_label_file,
)

SEED = 0
random.seed(SEED)
np.random.seed(SEED)


def main():
    """run LLM or rules to retrieve the sequences from the text

    pmc and abstract by LLM
    table by rules.
    """
    retriever_pmc = LLMRetriever(
        pickup=1,
        base_url="http://127.0.0.1:8001/",
        retrieve_abstract=False,
        input_file=pmc_input_file,
        test_input_file=pmc_test_split_file,
        test_results_dir=pmc_test_data_results_dir,
        test_label_file=pmc_test_split_label_file,
        question_template=FULL_EXAMPLES_QUESTION,
        result_base_filename=llama3_result_file_base,
        loops=8,
        overwrite=0,  # type: ignore
        save_iter=10,
    )
    # retriever_pmc.test_with_labels()
    retriever_pmc.retrieve()

    retriever_abstr = LLMRetriever(
        pickup=1,
        base_url="http://127.0.0.1:8001/",
        retrieve_abstract=True,
        input_file=abstract_input_file,
        test_input_file=abstract_test_file,
        test_label_file=abstract_test_file,  # type: ignore
        question_template=abstract_question,
        result_base_filename=abstarct_result_file_base,
        loops=8,
        overwrite=1,  # type: ignore
        save_iter=10,
    )
    # retriever_abstr.test_with_labels()
    # retriever_abstr.retrieve()


if __name__ == "__main__":
    main()