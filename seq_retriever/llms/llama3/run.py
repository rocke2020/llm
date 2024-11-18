import os
import sys

from loguru import logger

sys.path.append(os.path.abspath("."))
from seq_retriever.llms.questions_prompts import FULL_EXAMPLES_QUESTION, QUESTIONS
from seq_retriever.pmc.data_process import (
    llama3_result_file_base,
    pmc_input_file,
    pmc_input_non_english_file,
    pmc_test_split_file,
    pmc_test_split_label_file,
)
from seq_retriever.utils_comm.utils_retriever import (
    get_loop_num_and_generate_func,
    retrieve_seqs,
    set_rate_limit,
    test_prompts_with_query_func,
    update_pmc,
)
from utils_comm.file_util import file_util

BASE_URL = "http://127.0.0.1:8001/"


class LLMRetriever:
    """
    docstring
    """

    def __init__(
        self,
        model="llama3_8b",
        pickup=0,
        overwrite=False,
        base_url=BASE_URL,
        input_file=pmc_input_file,
        test_input_file=pmc_test_split_file,
        test_results_dir=None,
        test_label_file=pmc_test_split_label_file,
        question_template=FULL_EXAMPLES_QUESTION,
        retrieve_abstract=False,
        result_base_filename=llama3_result_file_base,
        loops=8,
        save_iter=10,
        test_num=0,
        all_papers_to_test=0,
    ) -> None:
        self.model = model
        self.pickup = pickup
        self.all_papers_to_test = all_papers_to_test
        self.overwrite = overwrite
        self.base_url = base_url
        self.question_template = question_template
        self.test_num = test_num
        self.rate_limit = set_rate_limit(self.model)
        self.generate_func, self.loops = get_loop_num_and_generate_func(
            self.model, loops, self.base_url
        )
        self.input_file = input_file
        self.test_input_file = test_input_file
        self.test_results_dir = test_results_dir
        self.test_label_file = test_label_file
        self.out_file = result_base_filename.with_stem(
            f"{result_base_filename.stem}_{self.model}_loops{self.loops}"
        )
        self.retrieve_abstract = retrieve_abstract
        self.save_iter = save_iter
        if self.test_results_dir is None:
            self.test_results_dir = input_file.parent
        self.test_results_dir.mkdir(exist_ok=True, parents=True)

    def test_with_labels(self):
        """
        If run picked test data, no need the label_file and not compare with labels.
        """
        if self.pickup:
            picked_test_data = {
                # PMC7052017 is a review of many peptides
                # PMC7023394 PMC7238586 PMC7052017 PMC10526274
                "pmc": "PMC7052017",
                "pmid": "24129228",
                # Introduction Results Methods
                # Historical Overview of CARPs and Neuroprotection Studies
                # CARPs Have Multimodal Neuroprotective Mechanisms Of Action_2
                # "section": "Introduction",
                # "section": "CARPs Have Multimodal Neuroprotective Mechanisms Of Action_1",
                "section": "Historical Overview of CARPs and Neuroprotection Studies",
            }
        else:
            picked_test_data = None
        if self.all_papers_to_test:
            input_file = self.input_file
        else:
            input_file = self.test_input_file
        pred_saved_file = (
            self.test_results_dir # type: ignore
            / f"{input_file.stem}_{self.model}_loops{self.loops}.json"
        )
        logger.info(f"{self.model = } {self.loops = } {self.base_url = }")
        test_prompts_with_query_func(
            self.generate_func,
            rate_limit=self.rate_limit,
            question_template=self.question_template,
            input_file=input_file,
            pred_saved_file=pred_saved_file,
            label_file=self.test_label_file,
            loops=self.loops,
            picked_test_data=picked_test_data,
            retrieve_abstract=self.retrieve_abstract,
            overwrite_pred_saved_file=self.overwrite,
            incl_abstract=False,
            run_num=0,
        )

    def test_prompts(self, questions: dict):
        logger.info(f"{self.model = } {self.loops = } {self.base_url = }")
        for question_name, question_template in questions.items():
            pred_saved_file = (
                self.test_results_dir # type: ignore
                / f"{self.test_input_file.stem}_{self.model}_loops{self.loops}-{question_name}.json"
            )
            test_prompts_with_query_func(
                self.generate_func,
                rate_limit=self.rate_limit,
                question_template=question_template,
                input_file=self.test_input_file,
                pred_saved_file=pred_saved_file,
                label_file=self.test_label_file,
                loops=self.loops,
                picked_test_data=None,
                retrieve_abstract=self.retrieve_abstract,
                overwrite_pred_saved_file=self.overwrite,
                incl_abstract=False,
            )

    def retrieve(self):
        input_data = file_util.read_json(self.input_file)
        logger.info(
            f"{self.model = } {self.loops = } {self.base_url = }"
            f"{self.input_file = } {self.out_file.stem = }, {self.test_num = }"
        )
        retrieve_seqs(
            self.question_template,
            input_data,
            self.out_file,
            self.generate_func,
            self.loops,
            self.rate_limit,
            self.test_num,
            overwrite=self.overwrite,
            retrieve_abstract=self.retrieve_abstract,
            save_iter=self.save_iter,
        )

    def update_partial_results(self, partial_updated_file=pmc_input_non_english_file):
        updated_input_data = file_util.read_json(partial_updated_file)
        logger.info(f"update_partial num, {len(updated_input_data) = }")
        logger.info(f"{self.out_file.stem = }")
        update_pmc(
            self.question_template,
            updated_input_data,
            self.out_file,
            self.generate_func,
            self.loops,
            self.rate_limit,
        )


if __name__ == "__main__":
    logger.info("start")
    retriever = LLMRetriever(
        model="llama3_8b",
        loops=8,
        pickup=0,
        all_papers_to_test=0,
        overwrite=1,  # type: ignore
        test_num=0,
    )
    retriever.test_with_labels()
    # test_abstract()
    logger.info("end")