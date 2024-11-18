import os
import re
import sys
from dataclasses import dataclass

from icecream import ic
from loguru import logger
from nltk.tokenize import sent_tokenize
from pandas import DataFrame
from transformers import AutoTokenizer

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120

sys.path.append(os.path.abspath("."))
from utils_comm.file_util import file_util

new_lines = re.compile(r"\n+")
MAX_TOKENS = 8192
MAX_NEW_TOKENS = 800
MAX_PROMPT_TOKENS = 2620
CONTEXT_WINDOW = MAX_TOKENS - MAX_NEW_TOKENS
MAX_INPUT_CONTENT_TOKENS = MAX_TOKENS - MAX_NEW_TOKENS - MAX_PROMPT_TOKENS
logger.info(f'\n{MAX_TOKENS = }, {MAX_NEW_TOKENS = }, {MAX_PROMPT_TOKENS = }, {MAX_INPUT_CONTENT_TOKENS = }')
"""
max length = 8192
max new token = 800
context_window = 7392
max template token ids as 2620
max input content token ids as 7392-2620=4772.
"""


use_info_log_level = 0
if use_info_log_level:
    logger.remove()  # remove the old handler.
    logger.add(sys.stdout, level="INFO")

TEST_TEXT = """What I Worked On

February 2021

Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays? I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep.

The first programs I tried writing were on the IBM 1401 that our school district used for what was then called "data processing." This was in 9th grade, so I was 13 or 14. The school district's 1401 happened to be in the basement of our junior high school, and my friend Rich Draves and I got permission to use it. It was like a mini Bond villain's lair down there, with all these alien-looking machines — CPU, disk drives, printer, card reader — sitting up on a raised floor under bright fluorescent lights.
"""


@dataclass
class Sentence:
    text: str
    token_num: int


@dataclass
class Part:
    """parts of text splitted from a long section text"""

    text: str
    token_num: int
    sent_indexes: list[int]


def sent_split(tokenizer: AutoTokenizer, text=TEST_TEXT, verbose=0) -> list[Sentence]:
    text = new_lines.sub("\n", text).strip()
    sections = text.split("\n")
    sentences_txt = []
    for section in sections:
        sentences_txt.extend(sent_tokenize(section))
        sentences_txt[-1] = sentences_txt[-1] + "\n"

    sentences = []
    sentence_token_nums = []
    for sent_text in sentences_txt:
        sentence_token_num = len(tokenizer.encode(sent_text))
        sentences.append(Sentence(sent_text, sentence_token_num))
        sentence_token_nums.append(sentence_token_num)
    df = DataFrame(sentence_token_nums, columns=["sentence_token_nums"])
    logger.info(f'\n{df.describe()}')
    if verbose:
        for i, sentence in enumerate(sentences):
            logger.info(f"{i}: {sentence}")
    return sentences


def split_text_into_smaller_parts(
    tokenizer: AutoTokenizer,
    text: str,
    token_overlap_num=270,
    verbose=0,
    sentence_overlap_max_num=5,
):
    """  
        sentence_token_nums
    count           202.000000
    mean             53.891089
    std              24.497497
    min              12.000000
    25%              38.000000
    50%              48.000000
    75%              65.000000
    max             192.000000    
    """

    all_token_num = len(tokenizer.encode(text))  # type: ignore

    if all_token_num <= MAX_INPUT_CONTENT_TOKENS:
        return [text]
    logger.debug(f"long para, {all_token_num = }")
    # head and tail overlap token num is 270, 54*5=270
    part_max_token_num = MAX_INPUT_CONTENT_TOKENS - token_overlap_num * 2
    sentences = sent_split(tokenizer, text, verbose=0)
    parts = []
    texts_in_current_part = []
    part_token_num = 0
    start_sent_i = 0
    end_sent_i_incl = 0
    for sent_i, sentence in enumerate(sentences):
        # Current sentence alone is too long
        # logger.debug(f"{sent_i = } {sentence.token_num = }")
        if sentence.token_num >= part_max_token_num:
            logger.info(f"single sentence token num > {part_max_token_num}. {sentence.text[:100] = } ")
            if texts_in_current_part:
                part_text = "".join(texts_in_current_part)
                parts.append(
                    Part(part_text, part_token_num, [start_sent_i, end_sent_i_incl])
                )
            # Directly treat this long sentence as a part
            parts.append(Part(sentence.text, sentence.token_num, [sent_i, sent_i]))
            part_token_num = 0
            texts_in_current_part = []
            start_sent_i = sent_i + 1
            end_sent_i_incl = sent_i + 1
        else:
            _part_token_num = part_token_num + sentence.token_num
            if _part_token_num <= part_max_token_num:
                part_token_num = _part_token_num
                texts_in_current_part.append(sentence.text)
                end_sent_i_incl = sent_i
            else:
                if texts_in_current_part:
                    part_text = " ".join(texts_in_current_part)
                    parts.append(
                        Part(part_text, part_token_num, [start_sent_i, end_sent_i_incl])
                    )
                part_token_num = sentence.token_num
                texts_in_current_part = [sentence.text]
                start_sent_i = sent_i
                end_sent_i_incl = sent_i

    if texts_in_current_part:
        part_text = " ".join(texts_in_current_part)
        parts.append(Part(part_text, part_token_num, [start_sent_i, end_sent_i_incl]))

    if verbose:
        for part in parts:
            logger.info(f"{part.token_num = }, {part.sent_indexes = }\n{part.text = }")
    parts_with_overlap_sentences = add_overlap_sentences(
        parts, sentences, MAX_INPUT_CONTENT_TOKENS, sentence_overlap_max_num, verbose
    )
    return [part.text for part in parts_with_overlap_sentences]


def add_overlap_sentences(
    parts: list[Part],
    sentences: list[Sentence],
    max_input_token_num: int,
    max_overlap_num: int,
    verbose: int = 0,
):
    parts_with_overlap_sentences = []
    for part in parts:
        orig_start_sent_i, oirg_end_sent_i_incl = part.sent_indexes
        start_sent_i = max(0, orig_start_sent_i - max_overlap_num)
        end_sent_i_incl = min(
            len(sentences) - 1, oirg_end_sent_i_incl + max_overlap_num
        )
        part_token_num_overlap = sum(
            [sentences[i].token_num for i in range(start_sent_i, end_sent_i_incl + 1)]
        )
        sentence_overlap_num = max_overlap_num
        while part_token_num_overlap > max_input_token_num:
            sentence_overlap_num = sentence_overlap_num - 1
            start_sent_i = max(0, orig_start_sent_i - sentence_overlap_num)
            end_sent_i_incl = min(
                len(sentences) - 1, oirg_end_sent_i_incl + sentence_overlap_num
            )
            # ic(sentence_overlap_num, start_sent_i, end_sent_i_incl)
            if sentence_overlap_num == 0:
                part_token_num_overlap = part.token_num
                break
            part_token_num_overlap = sum(
                [
                    sentences[i].token_num
                    for i in range(start_sent_i, end_sent_i_incl + 1)
                ]
            )
            # ic(part_token_num_overlap)

        part_text = " ".join(
            [sentences[i].text for i in range(start_sent_i, end_sent_i_incl + 1)]
        )
        parts_with_overlap_sentences.append(
            Part(part_text, part_token_num_overlap, [start_sent_i, end_sent_i_incl])
        )
    if verbose:
        for part in parts_with_overlap_sentences:
            logger.info(f"{part.token_num = }, {part.sent_indexes = }\n{part.text = }")

    return parts_with_overlap_sentences


def split_long_sections(data, saved_file):
    model_id = "/mnt/nas1/models/meta-llama/Meta-Llama-3-8B-Instruct"
    _tokenizer = AutoTokenizer.from_pretrained(model_id)
    for item in data:
        paragraph = item["paragraph"]
        long_sections = []
        section_split = {}
        for section_name, text in paragraph.items():
            texts = split_text_into_smaller_parts(_tokenizer, text, verbose=0)
            if len(texts) > 1:
                logger.info(f"{item['pmc'] = }, {section_name = }, {len(texts) = }")
                for i, part_text in enumerate(texts):
                    section_split[f"{section_name}_{i}"] = part_text
                long_sections.append(section_name)
        paragraph.update(section_split)
        for section_name in long_sections:
            del paragraph[section_name]
    file_util.write_json(data, saved_file)


if __name__ == "__main__":
    model_id = "/mnt/nas1/models/meta-llama/Meta-Llama-3-8B-Instruct"
    _tokenizer = AutoTokenizer.from_pretrained(model_id)
    # sent_split(_tokenizer, TEST_TEXT, verbose=1)
    # split_text_into_smaller_parts(
    #     _tokenizer, TEST_TEXT, max_token_num=63, token_overlap_num=15, verbose=1
    # )
    logger.info("end")