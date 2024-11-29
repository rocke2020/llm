import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.abspath("."))
from loguru import logger

from seq_retriever.pmc.data_process import (
    abstract_file,
    abstracts_dict_file,
    pmc_input_all_lang_file,
    pmc_input_file,
    pmc_input_file_v10,
    pmc_input_sections_filtered_file,
    pmc_inputs_raw_dir,
    pmc_merged_file,
    root_dir,
)
from seq_retriever.pmc.data_process.check_language import convert_all_to_eng
from seq_retriever.pmc.data_process.preprocess import convert_abstracts_to_dict
from seq_retriever.pmc.data_process.text_splitter import split_long_sections
from utils_comm.file_util import file_util

section_prefixes_excluded_file = (
    "seq_retriever/pmc/data_process/section_prefixes_excluded.txt"
)

root_sections = defaultdict(set)
section_pat = re.compile(r"^[\W\d]+|^[IVX]+\.\s+|[\W\d]+$")

word_pat = re.compile(r"[a-zA-Z]{2}")
max_section_length = 60
root_sections_too_long = defaultdict(set)


def add_text(section_dict, new_v, section):
    text = section_dict["text"]
    if word_pat.search(text):
        new_v[section].append(text)


def split_sections(
    pmc_inputs_raw_dir: Path,
    pmc_merged_file: Path,
    pmc_sections_filtered_file: Path,
    pmc_input_all_lang_file: Path,
    orig_file: Path,
    sections_json_file: Path,
    abstracts_dict_file: Path,
    abstr_file: Path,
    split_long=1,
):
    """
    use the root_section or not. It seems no use of root_section is better.

    current input format in "parsed_pmc_merged_raw.json":
    [
      {
        "section": "CBP conjugation enhances the efficacy of in a pulmonary fibrosis model",
        "text": "To explore whether the collagen-binding antibody engineering approach has versatile application for inflammatory diseases, we next conjugated CBP with (Fig. 4, B and C). These results suggest that CBP conjugation is applicable to other antibodies and that installing collagen affinity enables anti-inflammatory antibodies to target inflamed tissues and enhance its efficacy in inflammatory diseases and in fibrosis accompanied by inflammation.",
        "father_section": "RESULTS",
        "root_section": "RESULTS",
      },
      {
        "section": "DISCUSSION",
        "text": "Various drugs are in development based on passive targeting strategies to noncancer diseases using accumulation of macromolecular prodrugs, liposomes, and nanoparticles administered through the systemic circulation, although they are still in preclinical stages or suspended in early clinical stages (26\u201328). While these techniques are useful for delivering drugs to inflammatory sites, inflammation site targeting is challenging because of biological barriers such as ECM proteins and rapid clearance from the inflamed tissues via lymphatic drainage (15, 29). Immunocytokines have been targeted to the ECM spice variants fibronectin extra domain A (EDA) and extra domain B (EDB), which are expressed in sites of chronic inflammation (30). These targets are principally present in the inflamed subendothelium. Here, we target a majority component, collagen, which is present throughout the matrix of the inflamed tissue.",
        "father_section": "DISCUSSION",
        "root_section": "DISCUSSION",
      },
    """
    data = check_pmc_duplicate(orig_file)

    pmc_data = []
    kept_items = ["pmc", "pmid", "paragraph"]
    for item in data:
        paragraph = item["paragraph"]
        new_para = defaultdict(list)
        former_section = ""
        for i, section_dict in enumerate(paragraph):
            root_section = section_dict["root_section"]
            section = section_dict["section"]
            section = section_pat.sub("", section)
            if section:
                former_section = section
                add_text(section_dict, new_para, section)
                if len(root_section) > max_section_length:
                    root_sections_too_long[item["pmc"]].add(root_section)
                else:
                    root_sections[root_section.lower()].add(root_section)
            else:
                if i == 0 and former_section == "":
                    former_section = "Introduction"
                if former_section:
                    add_text(section_dict, new_para, former_section)
                else:
                    logger.warning(
                        f'{item["pmc"] = } has empty section at the first of paragraph. section text:\n{section_dict["text"][:100]}'
                    )

            # PMC6155558
            # if section.startswith('CGRP decreases corneal thickness, scar formation, and endothelial cell loss after injury'):
            #     logger.info(f'{item["pmc"] = } {section_dict["text"] = }')
            #     raise Exception
        item["paragraph"] = new_para

        new_item = {k: v for k, v in item.items() if k in kept_items}
        pmc_data.append(new_item)
    sections_txt_file = sections_json_file.with_suffix(".txt")
    with open(sections_txt_file, "w", encoding="utf-8") as f:
        for key in root_sections:
            f.write(f"{key}\n")
    file_util.write_json(root_sections, sections_json_file)
    sections_too_long_file = pmc_inputs_raw_dir / "sections_too_long.json"
    file_util.write_json(root_sections_too_long, sections_too_long_file)

    for item in pmc_data:
        pmc = item["pmc"]
        paragraph = item["paragraph"]
        new_para = {}
        for section, texts in paragraph.items():
            content = " ".join(texts)
            if len(content) < 10:
                logger.info(f"{pmc = } {section} is too short")
                continue
            # "Correspondence: Adel Galal El-Shemi (dr_adel_elshemy2006@yahoo.com) – Department of Pharmacology, Faculty of Medicine, Assiut University, Assiut, Egypt"
            if not section.startswith("Correspondence:"):
                content = section + "\n" + content
            new_para[section] = content
        item["paragraph"] = new_para
    file_util.write_json(pmc_data, pmc_merged_file)
    data_filtered = filter_sections(
        abstr_file, abstracts_dict_file, pmc_sections_filtered_file, pmc_data
    )
    if split_long:
        split_all_long_sections(pmc_input_all_lang_file, data=data_filtered)


def check_pmc_duplicate(orig_file):
    data = file_util.read_json(orig_file)
    pmc_ids = [item["pmc"] for item in data]
    assert len(pmc_ids) == len(set(pmc_ids))
    logger.info(f"{len(pmc_ids) = }")
    return data


def filter_sections(
    abstr_file, abstracts_dict_file, out_file, pmc_data=None, pmc_merged_file=None
):
    """
    pmc_merged_file -> pmc_input_file

    Input format:
        [
            {
                "pmc": "PMC7023394",
                "pmid": "31936124",
                "paragraph": {
                    "Introduction": "Peptides, polymers composed of amino acids,..."
                }
            },
        ]
    """
    abstracts = convert_abstracts_to_dict(abstr_file, abstracts_dict_file)
    if not pmc_data:
        pmc_data = file_util.read_json(pmc_merged_file)
    logger.info(f"len(pmc_data): {len(pmc_data)}")
    section_prefixes_excluded = file_util.read_lines_from_txt(
        section_prefixes_excluded_file
    )
    data_filtered = []

    for item in pmc_data:
        pmc = item.get("pmc", "none")
        if not "paragraph" in item:
            logger.warning(f"{pmc = } has no paragraph")
            continue
        paragraph = item["paragraph"]
        new_v = {}
        for sec, para in paragraph.items():
            for section_prefix in section_prefixes_excluded:
                if sec.lower().startswith(section_prefix):
                    break
            else:
                new_v[sec] = para
        item["paragraph"] = new_v
        if not new_v:
            logger.warning("%s has empty paragraph", pmc)
        pmid = item["pmid"]
        if pmid in abstracts:
            item["abstract"] = abstracts[pmid]["abstract"]
            item["title"] = abstracts[pmid]["title"]
        else:
            logger.warning(f"{pmc = } {pmid = } not in abstracts")
        data_filtered.append(item)
    file_util.write_json(data_filtered, out_file)
    return data_filtered


def split_all_long_sections(
    pmc_input_all_lang_file, data=None, pmc_input_sections_filtered_file=None
):
    if not data:
        data = file_util.read_json(pmc_input_sections_filtered_file)
    split_long_sections(data, pmc_input_all_lang_file)


def compare_version():
    """Compare pmc_input_file_v10 and pmc_input_file_v12

    len(old_pmc_ids) = 1486, len(new_pmc_ids) = 1984
    """
    old_file = pmc_input_file_v10
    new_file = pmc_input_sections_filtered_file
    old_data = file_util.read_json(old_file)
    new_data = file_util.read_json(new_file)
    old_pmc_ids = set([item["pmc"] for item in old_data])
    new_pmc_ids = set([item["pmc"] for item in new_data])
    logger.info(f"{len(old_pmc_ids) = }, {len(new_pmc_ids) = }")
    only_old_pmc_ids = old_pmc_ids - new_pmc_ids
    only_new_pmc_ids = new_pmc_ids - old_pmc_ids
    logger.info(f"{len(only_old_pmc_ids) = }")
    logger.info(f"{len(only_new_pmc_ids) = }")
    shared_pmc_ids = old_pmc_ids & new_pmc_ids
    logger.info(f"{len(shared_pmc_ids) = }")


if __name__ == "__main__":
    orig_pmc_file = pmc_inputs_raw_dir / "parsed_pmc_20240515.json"
    pmc_merged_file = pmc_inputs_raw_dir / "parsed_pmc_merged.json"
    sections_json_file = root_dir / "sections.json"

    split_sections(
        pmc_inputs_raw_dir,
        pmc_merged_file,
        pmc_input_sections_filtered_file,
        pmc_input_all_lang_file,
        orig_file=orig_pmc_file,
        sections_json_file=sections_json_file,
        abstracts_dict_file=abstracts_dict_file,
        abstr_file=abstract_file,
    )
    ## Must manually check the non-english judge, as LLM may make mistakes to treat englighs as non-english.
    convert_all_to_eng(
        pmc_input_all_lang_file, pmc_input_file, pmc_inputs_raw_dir, re_run=0
    )
    # compare_version()

    # check_pmc_duplicate(orig_pmc_file)

    # save_final_filterd_pmc()
    # split_all_long_sections()
    logger.info("end")