import os
import sys

from icecream import ic
from loguru import logger

sys.path.append(os.path.abspath("."))
from seq_retriever.test_data.creator import (
    create_fake_result_file,
    test_name_seqs,
    test_seqs,
)
from seq_retriever.utils_comm.result_merger import Merger
from seq_retriever.utils_comm.seq_parser import (
    postprocess_seq_and_name,
    valid_peptide_seq_pat,
    wrong_seq_pat,
)

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120


def test_without_content(results, task="anti_inflammatory_peptide"):
    """ """
    merger = Merger(
        # fake file in test
        pmc_result_file="seq_retriever/test_data/run_test.py",
        pmc_results=results,
        task=task,
        overwrite=1,
        save_result=False,
        merged_result_file="seq_retriever/test_data/run_test.py",
    )
    logger.warning(
        f"test to merge fake result from file: {results} with empty content!"
    )
    merger.merge_paragraph_pred()


def main():
    pred_results = create_fake_result_file(save_tmp_file=0)
    test_without_content(pred_results)  # type: ignore


def test_postprocess_peptide_seq():
    """ """
    # tests is a dict (pep_name, pep_seq)
    for orig_name, orig_seq in test_name_seqs:
        pep_name, pep_seq = postprocess_seq_and_name(orig_name, orig_seq)
        ic(orig_name, orig_seq, pep_name, pep_seq)


def test_peptide_seq_pattern(test_valid_seq=0, full_search_result=0):
    """
    DRS-DA2NEQ is peptide name, not seq.
    """
    if test_valid_seq:
        task = "test_valid_seq"
    else:
        task = "test_wrong_seq"
    print(f"{task = } !")
    for seq in test_seqs:
        if test_valid_seq:
            got = valid_peptide_seq_pat.search(seq)
        else:
            got = wrong_seq_pat.search(seq)
        if not full_search_result and got:
            got = "got"
        print(seq, got)


if __name__ == "__main__":
    main()
    logger.info("end")