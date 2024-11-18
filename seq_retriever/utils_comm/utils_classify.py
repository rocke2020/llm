import os
import sys
import time

import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm

sys.path.append(os.path.abspath("."))
from transformers import AutoTokenizer

from utils_comm.file_util import file_util
from utils_llama_index.model_comm import get_model_path

SYSTEM_PROMPT_BIO_PEPTIDE = """
You are a biologist, and your task is to judge whether or not the peptide sequences have some special function, based on the provided scientific article contents.
"""
system_input_bio = {"role": "system", "content": SYSTEM_PROMPT_BIO_PEPTIDE}


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
    MAX_PROMPT_TOKENS = 3100
    if len(input_ids) >= MAX_PROMPT_TOKENS:
        logger.error(f"{len(input_ids) = } > {MAX_PROMPT_TOKENS = }, is too long.")
        raise ValueError(f"{question[:100] = } is too long, advice to reduce the length.")


def classify_seqs(
    sequence_data,
    content_data,
    generate_func,
    out_file,
    question,
    loops,
    rate_limit,
    test_num,
    save_iter,
    overwrite,
    save_reply,
    class_from_abstract=False     
):
    logger.info(f"{os.path.isfile(out_file) = } {out_file = }")
    if not overwrite and os.path.isfile(out_file):
        logger.info(f"Load pred saved file {out_file}")
        pred_results = pd.read_csv(out_file)
        class_df = pred_results
        anti_seq = (pred_results['anti_inflammatory_pred'] == 1).sum()
        non_anti_seq = (pred_results['anti_inflammatory_pred'] == 0).sum()
        logger.info(f"In the pred saved file, {anti_seq = }, {non_anti_seq = }")
    else:
        class_df = sequence_data
        class_df["anti_inflammatory_pred"] = np.nan
    check_question_token_num(question)
    reply_file = out_file.with_stem(f"{out_file.stem}_reply")
    reply_file = reply_file.with_suffix(".json")
    reply_data = []
    if class_from_abstract:
        source_content = {item["pmid"]: item["abstract"] for item in content_data}
    else:
        source_content = {item["pmc"]: item["paragraph"] for item in content_data}
    for index in tqdm(range(len(class_df))):
        if not pd.isnull(class_df.loc[index, "anti_inflammatory_pred"]):
            logger.info(f"{index} has classifed")
            continue
        source = str(class_df['source'][index])
        logger.info(f"{source =}")
        peptide_seq = class_df["orig_seq"][index]
        contents = source_content[source]
        if class_from_abstract:
            anti_result, replies = query_by_loops_anti_seq_score(
                contents, peptide_seq, question, generate_func, loops, rate_limit, save_reply
            )
            if save_reply:
                reply_seq = {
                    "pmid": source,
                    "abstract": contents,
                    "peptide_seq": peptide_seq,
                    "llm_reply": replies,
                }
                reply_data.append(reply_seq)
            class_df.loc[index, "anti_inflammatory_pred"] = anti_result

        else:
            reply_seq = {
                "pmc": source,
                "paragraph": contents,
                "peptide_seq": peptide_seq,
            }
            paragraph_result = []
            for title, content in contents.items():
                anti_result, replies = query_by_loops_anti_seq_score(
                    content, peptide_seq, question, generate_func, loops, rate_limit, save_reply
                )
                if save_reply:
                    reply_seq[f"{title}_reply"] = replies
                paragraph_result.append(anti_result)
                if anti_result == 1:
                    class_df.loc[index, "anti_inflammatory_pred"] = 1
                    logger.info(f'find anti result and skip the following paragraphs')
                    break
            if save_reply:
                reply_data.append(reply_seq)
            if pd.isnull(class_df.loc[index, "anti_inflammatory_pred"]):
                class_df.loc[index, "anti_inflammatory_pred"] = 0

        if test_num > 0 and index + 1 == test_num:
            break
        if (index + 1) % save_iter == 0:
            class_df.to_csv(out_file, index=False)
    logger.info(f'"***** {out_file = }')
    if save_reply:
        file_util.write_json(reply_data, reply_file)
    class_df.to_csv(out_file, index=False)
    anti_seq_final = (class_df['anti_inflammatory_pred'] == 1).sum()
    non_anti_seq_final = (class_df['anti_inflammatory_pred'] == 0).sum()
    logger.info(f"In the result file, {anti_seq_final = }, {non_anti_seq_final = }")


def test_prompts_class_seq(
    sequence_data,
    content_data,
    query_func,
    out_file,
    question,
    loops=1,
    rate_limit=0,
    test_num=0,
    picked_test_data=None,
    save_iter=20,
    overwrite=True,
    save_reply=False,
    class_from_abstract=False     
):
    logger.info(f"{os.path.isfile(out_file) = } {out_file = }")
    if (
        not picked_test_data 
        and not overwrite 
        and os.path.isfile(out_file)
    ):
        logger.info(f"Load pred saved file {out_file}")
        pred_results = pd.read_csv(out_file)
        if not pred_results["anti_inflammatory_pred"].empty:
            calc_performance_anti_seq(pred_results)
            return
    check_question_token_num(question)
    reply_file = out_file.with_stem(f"{out_file.stem}_reply")
    reply_file = reply_file.with_suffix(".json")
    reply_data = []
    if class_from_abstract:
        source_content = {item["pmid"]: item["abstract"] for item in content_data}
    else:
        source_content = {item["pmc"]: item["paragraph"] for item in content_data}
    lables = sequence_data['anti_inflammatory_label']
    logger.info(f"{len(lables) = }, the number of anti is {sum(lables)}, the number of non-anti is {len(lables) - sum(lables)}")
    sequence_data["anti_inflammatory_pred"] = ''
    for index in tqdm(range(len(sequence_data))):
        source = str(sequence_data["source"][index])
        logger.info(f'"***** {index = }, {source = }')
        if picked_test_data:
            pick_content = None
            if class_from_abstract and source == picked_test_data['pmid']:
                logger.info(f' pick ***** {source = }')
                pick_content = source_content[source]
            elif not class_from_abstract and source == picked_test_data['pmc']:
                logger.info(f' pick ***** {source = }')
                paragraph = source_content[source]
                section_name = picked_test_data["section"]
                pick_content = paragraph[section_name]
            if pick_content:
                peptide_seq = sequence_data["peptide_seq"][index]
                anti_result, replies = query_by_loops_anti_seq_score(
                    contents, peptide_seq, question, query_func, loops, rate_limit, save_reply
                )
                break
        else:
            contents = source_content[source]
            peptide_seq = sequence_data["peptide_seq"][index]
            if class_from_abstract:
                anti_result, replies = query_by_loops_anti_seq_score(
                    contents, peptide_seq, question, query_func, loops, rate_limit, save_reply
                )
                if save_reply:
                    reply_seq = {
                        "pmid": source,
                        "abstract": contents,
                        "peptide_seq": peptide_seq,
                        "llm_reply": replies,
                    }
                    reply_data.append(reply_seq)
                sequence_data.loc[index, "anti_inflammatory_pred"] = anti_result

            else:
                reply_seq = {
                    "pmc": source,
                    "paragraph": contents,
                    "peptide_seq": peptide_seq,
                }
                paragraph_result = []
                have_anti = False
                for title, content in contents.items():
                    anti_result, replies = query_by_loops_anti_seq_score(
                        content, peptide_seq, question, query_func, loops, rate_limit, save_reply
                    )
                    if save_reply:
                        reply_seq[f"{title}_reply"] = replies
                    paragraph_result.append(anti_result)
                    if anti_result == 1:
                        sequence_data.loc[index, "anti_inflammatory_pred"] = 1
                        logger.info(f'find anti result and skip the following paragraphs ')
                        have_anti = True
                        break
                if save_reply:
                    reply_data.append(reply_seq)
                if not have_anti:   
                    sequence_data.loc[index, "anti_inflammatory_pred"] = 0
            if test_num > 0 and index + 1 == test_num:
                break
            if (index + 1) % save_iter == 0:
                sequence_data.to_csv(out_file, index=False)
            # file_util.write_json(reply_data, reply_file)
    logger.info(f'"***** {out_file = }')
    if save_reply:
        file_util.write_json(reply_data, reply_file)
    sequence_data.to_csv(out_file, index=False)
    calc_performance_anti_seq(sequence_data)


def query_by_loops_anti_seq_two(
    content, peptide_seq, question, query_func, loops=1, rate_limit=0, save_reply=False
):
    loop_result = []
    replies = []
    for i in range(loops):
        input_content = question.format(content=content, peptide_seq=peptide_seq)
        temperature = 0.1 * i
        # logger.info(f"{temperature = }")
        if i == 0:
            logger.info(f"{peptide_seq = }, {input_content = }")
        reply = query_func(input_content, temperature=temperature)
        if "yes" in reply.lower():
            loop_result.append(1)
        elif "no" in reply.lower():
            loop_result.append(0)
        else:
            loop_result.append(0.5)
        logger.info(f"llm_reply {i+1} times\n{reply}")
        if save_reply:
            replies.append(reply)
        # online api has rate limit, so sleep 5s.
        if rate_limit:
            time.sleep(rate_limit)
    average_score = sum(loop_result) / len(
        loop_result
    )  # the finall result is the average
    logger.info(f"{average_score = }")
    if average_score > 0.5:
        final_result = 1
    else:
        final_result = 0
    return final_result, replies

def query_by_loops_anti_seq_score(
    content, peptide_seq, question, query_func, loops=1, rate_limit=0, save_reply=False
):
    loop_score = []
    replies = []
    score = 0
    for i in range(loops):
        input_content = question.format(content=content, peptide_seq=peptide_seq)
        if i == 0:
            logger.info(f"{peptide_seq = }, {input_content = }")
        temperature = 0.1 * i
        # logger.info(f"{temperature = }")
        reply = query_func(input_content, temperature=temperature)
        logger.info(f"llm_reply {i+1} times\n{reply}")
        lines = reply.split("\n")
        for line in lines:
            if line.startswith("* Anti-inflammatory score"):
                parts = line.split(":", maxsplit=1)
                score = parts[1].strip()
                if score == 'N/A':
                    loop_score.append(0.0)
                else:
                    loop_score.append(float(score))
        if len(loop_score) == 0:
            loop_score.append(score)
        logger.info(f"{score = }")
        if save_reply:
            replies.append(reply)
        # online api has rate limit, so sleep 5s.
        if rate_limit:
            time.sleep(rate_limit)
    average_score = sum(loop_score) / len(loop_score)  # the finall result is the average
    logger.info(f"{average_score = }")
    if average_score >= 80:
        final_result = 1
    else:
        final_result = 0
    return final_result, replies


def calc_performance_anti_seq(df_seq):
    anti_label = df_seq["anti_inflammatory_label"]
    anti_pred = df_seq["anti_inflammatory_pred"]
    tp = sum(x == 1 and y == 1 for x, y in zip(anti_pred, anti_label))
    fp = sum(x == 1 and y == 0 for x, y in zip(anti_pred, anti_label))
    fn = sum(x == 0 and y == 1 for x, y in zip(anti_pred, anti_label))

    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if (fn + tp) == 0:
        recall = 0
    else:
        recall = tp / (fn + tp)
    logger.info(f"{precision = } {recall = }, {tp = }, {fp = }, {fn = }")


if __name__ == "__main__":
    # from utils_llama_index.api_client import chat_with_context
    # data_postprocee()
    logger.info("end")