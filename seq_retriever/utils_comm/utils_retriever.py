import os
import shutil
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

from loguru import logger
from tqdm import tqdm

sys.path.append(os.path.abspath("."))
from transformers import AutoTokenizer

from seq_retriever.llms.questions_prompts import FULL_EXAMPLES_QUESTION
from seq_retriever.pmc.data_process.text_splitter import MAX_PROMPT_TOKENS
from seq_retriever.utils_comm.seq_parser import (
    UNAVAILABLE,
    parse_reply,
    postprocess_seq_and_name,
    process_peptide_seq,
)
from utils_comm.file_util import file_util
from utils_llama_index.api_client import vllm_generate_with_tokenizer
from utils_llama_index.model_comm import get_model_path

SYSTEM_PROMPT_BIO_PEPTIDE = """
You are a biologist, and your task is to extract and list peptide sequences from provided scientific article contents.
"""
system_input_bio = {"role": "system", "content": SYSTEM_PROMPT_BIO_PEPTIDE}
questions = [
    # QUESTIONS["concise_question"],
    # QUESTIONS["basic_examples_question"],
    # QUESTIONS["more_examples_question"],
    # QUESTIONS["full_examples_question"],
    FULL_EXAMPLES_QUESTION,
]
PARAGRAPH_PRED = "paragraph_pred"


def get_tokenizer(model_name="Llama-3"):
    model_path = get_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def check_question_token_num(question):
    tokenizer = get_tokenizer()
    logger.info(f'{question[:300] = }')
    logger.info(f'{question[-300:] = }')
    messages = [system_input_bio, {"role": "user", "content": question}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True)
    logger.info(f"{len(input_ids) = }")
    if len(input_ids) >= MAX_PROMPT_TOKENS:
        logger.error(f"{len(input_ids) = } > {MAX_PROMPT_TOKENS = }, is too long.")
        raise ValueError(f"{question[:100] = } is too long, advice to reduce the length.")


def test_prompts_with_query_func(
    query_func,
    rate_limit,
    question_template,
    input_file,
    pred_saved_file,
    label_file,
    loops=1,
    overwrite_pred_saved_file=1,
    picked_test_data=None,
    run_num=0,
    incl_abstract=False,
    retrieve_abstract=False,
):
    """If run picked test data, no need the label_file and not compare with labels.

    Args:
        query_func: Inputs is (question, context); outputs is str of llm_reply.
        run_num: <=0, run all test articles.
        incl_abstract: not included is better.
    """

    if (
        not picked_test_data
        and not overwrite_pred_saved_file
        and os.path.isfile(pred_saved_file)
    ):
        logger.info(f'Loads pred saved file {pred_saved_file}')
        pred_results = file_util.read_json(pred_saved_file)
        if pred_results:
            calc_performance(pred_results, label_file, retrieve_abstract, show_recall=0, show_precision=1)
            return
    check_question_token_num(question_template)
    # sys.exit()
    data = file_util.read_json(input_file)
    logger.info(f'Test {len(data) = }')
    # logger.info(f"***** {question_template = }")
    pred_results = []
    for item_i, article in enumerate(data):
        pmc = article.get("pmc", '')
        pmid = article["pmid"]
        if picked_test_data:
            is_picked_data = False
            if not retrieve_abstract and pmc == picked_test_data["pmc"]:
                logger.info(f' pick ***** {pmc = }')
                paragraph = article["paragraph"]
                section_name = picked_test_data["section"]
                content = paragraph[section_name]
                is_picked_data = True
            elif retrieve_abstract and pmid == picked_test_data["pmid"]:
                logger.info(f' pick ***** {pmid = }')
                content = article["abstract"]
                is_picked_data = True
            if is_picked_data:
                input_txt = question_template.format(content=content)
                section_result, replies = query(
                    input_txt, query_func, loops, rate_limit
                )
                break
        else:
            logger.info(f'{item_i = }, {pmc = }, {pmid = }')
            if retrieve_abstract:
                abstr_pred, abstr_replies = parse_abstr(
                    article,
                    question_template,
                    query_func,
                    loops,
                    rate_limit,
                    pmid,
                )
                pred_results.append({"pmc": pmc, "pmid": pmid, "abstract_pred": abstr_pred})
            else:
                paragraph_pred, paragraph_replies = parse_paragraph(
                    article,
                    question_template,
                    query_func,
                    loops,
                    rate_limit,
                    incl_abstract=incl_abstract,
                )
                pred_results.append({"pmc": pmc, "pmid": pmid, "paragraph_pred": paragraph_pred})
            if run_num > 0 and item_i + 1 == run_num:
                break

    if not picked_test_data:
        file_util.write_json(pred_results, pred_saved_file)
        logger.info(f'Saves test with lablel results to {pred_saved_file = }')
        calc_performance(pred_results, label_file, retrieve_abstract, show_recall=0, show_precision=1)


def parse_paragraph(
    article_data,
    template,
    query_func,
    loops,
    rate_limit,
    incl_abstract=False,
):
    paragraph = article_data["paragraph"]
    abstract = article_data.get("abstract", "")
    title = article_data.get("title", "")
    logger.info(f"{title = }")
    if not abstract:
        incl_abstract = 0
    paragraph_pred = defaultdict(dict)
    paragraph_replies = defaultdict(list)
    for section_name, content in paragraph.items():
        logger.info(f"{section_name}: {content}")
        if incl_abstract:
            content = abstract + "\n" + content
        input_txt = template.format(content=content)
        section_result, replies = query(input_txt, query_func, loops, rate_limit)
        paragraph_pred[section_name] = section_result
        paragraph_replies[section_name] = replies
    return paragraph_pred, paragraph_replies


def parse_abstr(
    article_data,
    question,
    query_func,
    loops,
    rate_limit,
    pmid,
):
    title = article_data.get("title", "")
    logger.info(f"{title = }")
    if pmid == '27454254':   # the abstract is too long
        return {}, []
    abstract = article_data.get("abstract", "")
    input_txt = question.format(content=abstract)
    section_result, replies = query(input_txt, query_func, loops, rate_limit)
    return section_result, replies


def query(input_txt, query_func, loops=1, rate_limit=0):
    """Auto log out the query(generate from LLM) result for each loop.

    Args:
        loop: loop_num, default 1, test one question with loops times for randomness check.
    Return:
        only the reply from the first loop.
    """
    section_result = {}
    replies = []
    for i in range(loops):
        temperature = 0.1 * i
        reply = query_func(input_txt, temperature=temperature)
        result = parse_reply(reply)
        logger.info(f"llm_reply {i + 1} times\n{reply}\n{result = }")
        for peptide_name, peptide_seqs in result.items():
            former_set = section_result.get(peptide_name, set())
            section_result[peptide_name] = former_set.union(peptide_seqs)
        # online api has rate limit, so sleep 5s.
        if rate_limit:
            time.sleep(rate_limit)
        replies.append(reply)
    logger.info(f"merged {section_result = }")
    for pep_name, pep_seqs in section_result.items():
        section_result[pep_name] = sorted(pep_seqs)
    return section_result, replies


def get_generate_func(model, base_url):
    """
    Args:
        base_url: default, "http://127.0.0.1:8001/", we can use 8008 etc.
    """
    if model == "llama3_8b":
        generate_func = partial(vllm_generate_with_tokenizer, base_url=base_url)
    else:
        from utils_llama_index.api_online import llama3_70b_generate, llama3_sonar

        if model == "llama3_70b":
            generate_func = llama3_70b_generate
        elif model == "llama3_sonar":
            generate_func = llama3_sonar
    return generate_func


def get_loop_num_and_generate_func(model, loop_num, base_url):
    if model != "llama3_8b":
        loop_num = 1
    generate_func = get_generate_func(model, base_url)
    return generate_func, loop_num


def calc_performance(pred_results, label_file, retrieve_abstract, show_recall=0, show_precision=1):
    """
    label_data example:
    [
        {
        "pmc": "PMC7023394",
        "pmid": "31936124",
        "paragraph_pred": {
            "Introduction": {
                "copper glycine-histidine-lysine": [
                    "Cu-GHK"
                ],
                "tetrapeptide PKEK": [
                    "PKEK"
                ],
                "CPPAIF": [
                    "None"
                ],
        ]
    """
    label_data = file_util.read_json(label_file)
    tp = 0
    fp = 0
    fn_tp = 0
    for label_data, pred_data in zip(label_data, pred_results):
        if retrieve_abstract:
            pmid_data = label_data["pmid"]  # some abstract dont have pmc
            pmid_pred = pred_data["pmid"]
            assert pmid_data == pmid_pred, f"{pmid_data = } != {pmid_pred = }"
            logger.info(f"pmid {pmid_data}")
            abstract_labels = label_data["labels"]
            content = label_data["abstract"]
            abstract_preds = pred_data["abstract_pred"]
            section = 'abstract'
            tp, fp, fn_tp = calc_section(
                tp,
                fp,
                fn_tp,
                content,
                section,
                abstract_labels,
                abstract_preds,
                show_recall,
                show_precision,
            )
        else:
            pmc_data = label_data["pmc"]
            pmc_pred = pred_data["pmc"]
            assert pmc_data == pmc_pred, f"{pmc_data = } != {pmc_pred = }"
            logger.info(f"pmc {pmc_data}")
            paragragh_labels = label_data["labels"]
            paragraph_preds = pred_data[PARAGRAPH_PRED]
            for section, section_label_dict in paragragh_labels.items():
                if section not in paragraph_preds:
                    logger.warning(f"{section = } not in llm_predictions")
                    continue
                section_pred_dict = paragraph_preds[section]
                content = label_data["paragraph"][section]
                tp, fp, fn_tp = calc_section(
                    tp,
                    fp,
                    fn_tp,
                    content,
                    section,
                    section_label_dict,
                    section_pred_dict,
                    show_recall,
                    show_precision,
                )
    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if fn_tp == 0:
        recall = 0
    else:
        recall = tp / fn_tp
    logger.info(f"{precision = } {recall = }, {tp = }, {fp = }, {fn_tp = }")


def calc_section(
    tp,
    fp,
    fn_tp,
    content,
    section,
    section_label_dict,
    section_pred_dict,
    show_recall,
    show_precision,
):
    para_post_preds = {}
    for peptide_name, peptide_seq_list in section_pred_dict.items():
        for i, peptide_seq in enumerate(peptide_seq_list):
            pep_name, peptide_seq = postprocess_seq_and_name(
                peptide_name, peptide_seq, content
            )
            if i > 0:
                peptide_name = f"{peptide_name}_{i}"
            para_post_preds[peptide_name] = peptide_seq
    non_empty_seq_label_dict = {
        v.strip(): k.lower() for k, v in section_label_dict.items() if v != "None"
    }
    logger.info(f"{non_empty_seq_label_dict = }")
    non_empty_seq_pred_dict = {
        v.strip(): k.lower()
        for k, v in para_post_preds.items()
        if v not in ("None", UNAVAILABLE)
    }
    true_seqs = set(non_empty_seq_label_dict.keys())
    pred_seqs = set(non_empty_seq_pred_dict.keys())
    tp += len(true_seqs & pred_seqs)
    fp += len(pred_seqs - true_seqs)
    fn_tp += len(true_seqs)
    if show_precision:
        for peptide_seq, peptide_name in non_empty_seq_pred_dict.items():
            if peptide_seq not in true_seqs:
                logger.info(
                    f"'Precision': {section = }, pred {peptide_name = }, "
                    f"pred {peptide_seq = }, not in true_seqs {true_seqs = }"
                )
    if show_recall:
        for peptide_seq, peptide_name in non_empty_seq_label_dict.items():
            if peptide_seq not in pred_seqs:
                logger.info(
                    f"'Recall': {section = }, true {peptide_name = }, "
                    f"true {peptide_seq = }, not in pred_seqs {pred_seqs = }"
                )
    return tp, fp, fn_tp


def retrieve_seqs(
    question,
    input_data,
    out_file,
    query_func,
    loops,
    rate_limit=0,
    test_num=0,
    overwrite=True,
    incl_abstract=False,
    retrieve_abstract=False,
    save_iter=20,
):
    """
    Args:
        run_num: int, default 1, the number of samples to query. If <= 0, test all samples.
        reverse_input_data: int, default 0, if 1, reverse the input data order.
    """
    check_question_token_num(question)
    out_file = Path(out_file)
    if not overwrite and out_file.exists():
        pred_results = file_util.read_json(out_file)
        logger.info(f"loaded pre pre_retrieved_pmc {len(pred_results) = }")
        if retrieve_abstract:
            pre_retrieved_pmc_ids = [item["pmid"] for item in pred_results]
        else:
            pre_retrieved_pmc_ids = [item["pmc"] for item in pred_results]
    else:
        pre_retrieved_pmc_ids = []
        pred_results = []
    for i, article_data in enumerate(tqdm(input_data)):
        pmc = article_data.get("pmc", "")
        pmid = article_data["pmid"]
        if retrieve_abstract and pmid in pre_retrieved_pmc_ids:
            logger.info(f"{pmid = } pre-retrieved, and skip")
            continue
        if not retrieve_abstract and pmc in pre_retrieved_pmc_ids:
            logger.info(f"{pmc = } pre-retrieved, and skip")
            continue
        logger.info(f"{pmc = }, {pmid = }, {i = }")
        result = {"pmc": pmc, "pmid": pmid, "title": article_data.get("title", "")}
        if retrieve_abstract:
            logger.info("retrieve seq from abstract")
            abstr_pred, llm_replies = parse_abstr(
                article_data,
                question,
                query_func,
                loops,
                rate_limit,
                pmid,
            )
            result["abstract_pred"] = abstr_pred
        else:
            paragraph_pred, llm_replies = parse_paragraph(
                article_data,
                question,
                query_func,
                loops,
                rate_limit,
                incl_abstract=incl_abstract,
            )
            result[PARAGRAPH_PRED] = paragraph_pred
        result["llm_replies"] = llm_replies
        pred_results.append(result)

        if test_num > 0 and i >= test_num:
            file_util.write_json(pred_results, out_file)
            break
        if (i + 1) % save_iter == 0:
            file_util.write_json(pred_results, out_file)
        logger.info(f"{len(pred_results) = }")

    file_util.write_json(pred_results, out_file)


def set_rate_limit(model):
    if model == "llama3_8b":
        rate_limit = 0
    # online api needs rate limit
    else:
        rate_limit = 5
    return rate_limit


def update_pmc(question, input_data, out_file: Path, query_func, loop, rate_limit):
    if not out_file.is_file():
        logger.error(f"{out_file = } not exists as a file.")
        return

    if not input_data:
        logger.error(f"{input_data = } is empty.")
        return

    pred_results = file_util.read_json(out_file)
    logger.info(f"loaded pre pre_retrieved_pmc {len(pred_results) = }")
    pre_retrieved_pmc_ids = [item["pmc"] for item in pred_results]
    shutil.copy(out_file, out_file.with_stem(f"{out_file.stem}_bak"))

    for i, article_data in enumerate(tqdm(input_data)):
        pmc = article_data["pmc"]
        if pmc not in pre_retrieved_pmc_ids:
            logger.warning(f"{pmc = } not in pre-retrieved, and skip")
            continue
        logger.info(f"{pmc = } is updated")
        paragraph_pred, paragraph_replies = parse_paragraph(
            article_data, question, query_func, loop, rate_limit
        )
        current_pmc_idx = pre_retrieved_pmc_ids.index(pmc)
        pred_results[current_pmc_idx][PARAGRAPH_PRED] = paragraph_pred

    file_util.write_json(pred_results, out_file)


if __name__ == "__main__":
    # from utils_llama_index.api_client import chat_with_context
    # check_question_token_num()

    logger.info("end")