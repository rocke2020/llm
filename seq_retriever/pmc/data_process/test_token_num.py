import os
import sys

import torch
from icecream import ic
from loguru import logger
from nltk import word_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120

sys.path.append(os.path.abspath("."))
from seq_retriever.llms.questions_prompts import QUESTIONS

from seq_retriever.pmc.data_process import pmc_input_file_v10, pmc_test_file
from seq_retriever.pmc.data_process.text_splitter import CONTEXT_WINDOW
from utils_comm.file_util import file_util

model_id = "/mnt/nas1/models/meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

SYSTEM_PROMPT_BIO_PEPTIDE = """
You are a biologist, and your task is to extract and list peptide sequences from provided scientific article contents.
"""

test_str = """
Here is the list of peptide names and their corresponding sequences:

* LQLDEETGEFLPIQ: LQLDEETGEFLPIQ
* TAT-14: YGRKKRRQRRR-LQLDEETGEFLPIQ
* 7R-ETGE: RRRRRRRR-LQLDEETGEFLPIQ
* TAT-CAL-DEETGE: Table 2
* E5: Ac-SHLRKLRKRLLRDADDKRLA-NH2
* Ac-hE18A-NH2: Ac-RKLRKRLLRDWLKAFYDKVAEKLKEAF-NH2
* COG-133: Ac-LRVRLASHLRKLRKRLLR-NH2
* COG112: COG133 fused to the CCPP penetratin (Table 2)
* PACAP38: Table 3
* LL-37: LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES
* dRK: rrkrrr
* IG-19: IGKEFKRIVQRIKDFLRNL-NH2
* IDR-1018: Table 1
* TAT-BH3: Ac-RKKRR-O-RRR-EIWIAQELRRIGDEFNAYYAR
* R9-SOCS1-KIR: RRRRRRRRR-DTHFRTFRSHSDYRRI
* R9D: Table 1
* hBD3-3: GKCSTRGRKCCRRKK
* Protamine: Table 1
"""

# ids = tokenizer.encode(test_str)
# ic(len(ids), ids[-10:], tokenizer.decode(ids[-10:]))

messages = [
    {"role": "system", "content": SYSTEM_PROMPT_BIO_PEPTIDE},
    {"role": "user", "content": ""},
]


def calc_template_token_num():
    question = QUESTIONS["full_examples_question"]
    messages[-1]["content"] = question.format(content="")
    template_input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=True
    )
    ic(len(template_input_ids))


def calc_query_token_num(only_test_file=1):
    """input_id_len / len(words) = 1.42"""
    question = QUESTIONS["full_examples_question"]
    logger.info(f"question template\n{question}")
    messages[-1]["content"] = question.format(content="")
    template_input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=True, return_tensors="pt"
    )
    template_input_ids_len = len(template_input_ids[0])
    max_input_conent_ids_len = CONTEXT_WINDOW - template_input_ids_len
    ic(template_input_ids_len, max_input_conent_ids_len)

    if only_test_file:
        data = file_util.read_json(pmc_test_file)
    else:
        data = file_util.read_json(pmc_input_file_v10)
    ic(len(data))
    for i, item in enumerate(data):
        pmc = item["pmc"]
        # logger.info(f"{pmc = }")
        paragraph = item["paragraph"]
        for section_name, content in paragraph.items():
            messages[-1]["content"] = question.format(content=content)
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
                return_tensors="pt",
            )
            input_id_len = len(input_ids[0])
            if input_id_len > CONTEXT_WINDOW:
                logger.warning(f"{pmc = }, {section_name = }, {input_id_len = }")
                messages_str = []
                for message in messages:
                    for role, content in message.items():
                        messages_str.append(f"{role}: {content}")
                messages_str = "\n".join(messages_str)
                # logger.warning(f"{messages_str}")
                words = word_tokenize(messages_str)
                logger.info(f"{len(words) = }, {input_id_len / len(words) = }")


if __name__ == "__main__":
    # calc_query_token_num()
    calc_template_token_num()