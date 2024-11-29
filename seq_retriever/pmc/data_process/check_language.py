import os
import shutil
import sys
from collections import defaultdict

from loguru import logger

sys.path.append(os.path.abspath("."))
from seq_retriever.pmc.data_process import (
    pmc_input_all_lang_file,
    pmc_input_file,
    pmc_inputs_raw_dir,
)
from utils_comm.file_util import file_util
from utils_llama_index.api_client import vllm_generate_with_tokenizer

system_language_expert = {"role": "system", "content": "You are a language_expert."}

question_template = (
    "input content is below.\n---------------------\n"
    "{context}\n--------------------\n"
    'please judge the language is English or not, output "Yes" or "No". If the input content not English, do not output extra "Translation:" in the next line, but just translate it to English in the next line.'
)
PARAGRAPH = "paragraph"
other_main_sections = ["abstract", "title"]
all_main_sections = [
    "paragraph",
    "abstract",
    "title",
]


def check_language(all_lang_file, inputs_raw_dir, verbose=0, pick_pmc=0):
    """check language of each section in pmc_input_file
    TODO loop 7 times with 0.5 temparature and choose the higher possibility. LLM is not always right. Also check section name language.
    """
    pmc_input_non_english_file = inputs_raw_dir / "pmc_inputs_non_english.json"
    save_iter = 20
    orig_data = file_util.read_json(all_lang_file)
    non_english_pmc_ids = []
    all_translated_english_data = []
    total_num = len(orig_data)
    for article_i, article_data in enumerate(orig_data):
        pmc = article_data.get("pmc", "")
        if pick_pmc and pmc != "PMC10799709":
            continue
        logger.info(f'{article_i = } {total_num = }, {pmc = }')
        paragraph = article_data.get(PARAGRAPH, "")
        is_non_english = False
        translated_english_data = {}
        paragraph_en = {}
        not_log_non_english = True
        for i, (section_name, content) in enumerate(paragraph.items()):
            question = question_template.format(context=content)
            response = vllm_generate_with_tokenizer(
                question, system_input=system_language_expert
            )
            if verbose:
                logger.info(f"{section_name}, response\n{response}")
            if response.startswith("No\n"):
                is_non_english = True
                translated_english = response[3:].strip()
                paragraph_en[section_name] = translated_english
            else:
                paragraph_en[section_name] = content
            if i > 0 and not is_non_english:
                logger.info(f'English {content[:100] = }')
                break
            if is_non_english and not_log_non_english:
                not_log_non_english = False
                logger.info(f"{pmc = } {section_name} Non-English {content[:100] = }")

        if is_non_english:
            translated_english_data["pmc"] = pmc
            translated_english_data["pmid"] = article_data.get("pmid", "")
            translated_english_data[PARAGRAPH] = paragraph_en
            non_english_pmc_ids.append(pmc)
            for main_section_name in other_main_sections:
                content = article_data.get(main_section_name, "")
                if content:
                    question = question_template.format(context=content)
                    response = vllm_generate_with_tokenizer(
                        question, system_input=system_language_expert
                    )
                    if response.startswith("No\n"):
                        translated_english = response[3:].strip()
                        translated_english_data[main_section_name] = translated_english
            for main_section in all_main_sections:
                translated_english_data[f"{main_section}_orig"] = article_data.get(
                    main_section, ""
                )
            all_translated_english_data.append(translated_english_data)
        if (article_i + 1) % save_iter == 0:
            logger.info(f"{len(non_english_pmc_ids) = }")
            file_util.write_json(all_translated_english_data, pmc_input_non_english_file)
    logger.info(f"{len(non_english_pmc_ids) = }")
    file_util.write_json(all_translated_english_data, pmc_input_non_english_file)


def replace_non_english_paragraph(all_lang_file, all_eng_file, inputs_raw_dir):
    """ LLM make wrong judge english paper as non-english and so manually check and add excluded items """
    # LLM wrongly judge these two as non-english
    exclu_english_pmc_ids = ["PMC2241586", "PMC10799709", 'PMC6732857']
    orig_data = file_util.read_json(all_lang_file)
    logger.info(f"{len(orig_data) = }")
    pmc_input_non_english_file = inputs_raw_dir / "pmc_inputs_non_english.json"
    all_translated_english_data = file_util.read_json(pmc_input_non_english_file)
    new_data = []

    for article_data in orig_data:
        pmc = article_data.get("pmc", "")
        if pmc in exclu_english_pmc_ids:
            translated_english_data = None
        else:
            translated_english_data = next(
                (x for x in all_translated_english_data if x["pmc"] == pmc), None
            )
        if translated_english_data:
            logger.info(f'{pmc = } is Not English')
            new_data.append(translated_english_data)
        else:
            new_data.append(article_data)
    assert len(new_data) == len(orig_data)
    file_util.write_json(new_data, all_eng_file)


def convert_all_to_eng(all_lang_file, all_eng_file, inputs_raw_dir, re_run=0):
    if re_run:
        check_language(all_lang_file, inputs_raw_dir, verbose=0, pick_pmc=0)
    replace_non_english_paragraph(all_lang_file, all_eng_file, inputs_raw_dir)


if __name__ == "__main__":
    # check_language(verbose=0, pick_pmc=1)
    # replace_non_english_paragraph()
    convert_all_to_eng(
        pmc_input_all_lang_file, pmc_input_file, pmc_inputs_raw_dir, re_run=1
    )
    logger.info("end")