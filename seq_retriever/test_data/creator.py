import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from icecream import ic
from loguru import logger

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120
TEST_FILE = "seq_retriever/test_data/peptide_name_and_seqs.csv"
FAKE_RAW_PRED_FILE = Path("seq_retriever/test_data/fake_raw_pred.json")
NAME = "Name"
SEQUENCE = "Sequence"
NOTES = "Notes"
SEQ_PROCESSED_LABEL = "seq_postprocessed_label"
SEQ_NORMALIZED_LABEL = "seq_normalized_label"

# three columns: Name,Sequence,Notes
test_name_seqs_df = pd.read_csv(TEST_FILE)
name_is_not_null = test_name_seqs_df[[NAME]].notnull()
assert len(test_name_seqs_df) == len(name_is_not_null)
test_name_seqs_df.fillna("None", inplace=True)
test_name_seqs = test_name_seqs_df[[NAME, SEQUENCE]].fillna("None").values.tolist()
tmp_seq_and_notes_file = Path("seq_retriever/test_data/z.seqs.csv")
test_seqs = list(set([seq for name, seq in test_name_seqs]))


def create_fake_result_file(save_tmp_file=0):
    article_results = {
        "pmc": "PMC7023394",
        "pmid": "31936124",
        "title": "Cell Penetrating Peptide as a High Safety Anti-Inflammation Ingredient for Cosmetic Applications.",
        "paragraph_pred": {
            "Introduction": {},
            "Methods": {},
        },
    }

    introduction_num = int(len(test_seqs) / 2)
    logger.info(f"{len(test_name_seqs_df) = }")
    para_postprocessed = defaultdict(dict)
    para_normalized = defaultdict()
    seqs_postprocessed = []
    seqs_normalized = []
    for i, row in test_name_seqs_df.iterrows():
        i = int(i)  # type: ignore
        name = row[NAME]
        seq = row[SEQUENCE]
        seq_postprocessed_label = row[SEQ_PROCESSED_LABEL]
        seq_normalized_label = row[SEQ_NORMALIZED_LABEL]

        if i < introduction_num:
            article_results["paragraph_pred"]["Introduction"][name] = seq
            para_postprocessed["Introduction"][name] = seq_postprocessed_label
        else:
            article_results["paragraph_pred"]["Methods"][name] = seq
            para_postprocessed["Methods"][name] = seq_postprocessed_label

        if seq_postprocessed_label != 'None' and seq_postprocessed_label in seqs_postprocessed:
            logger.warning(f'row {i+2} has duplicate seq_postprocessed_label: {seq_postprocessed_label}')
            continue
        seqs_postprocessed.append(seq_postprocessed_label)

        is_unique_seq_normalized = seq_normalized_label not in seqs_normalized
        if name not in para_normalized and (
            seq_normalized_label == "None" or is_unique_seq_normalized
        ):
            para_normalized[name] = seq_normalized_label
        else:
            if name in para_normalized:
                logger.warning(f"row {i+2} has duplicate pep name {name}!")
            if not is_unique_seq_normalized:
                logger.warning(f"row {i+2} {is_unique_seq_normalized = }")
        seqs_normalized.append(seq_normalized_label)

    article_results[SEQ_PROCESSED_LABEL] = para_postprocessed
    article_results[SEQ_NORMALIZED_LABEL] = para_normalized
    total_saved_num = len(article_results["paragraph_pred"]["Introduction"]) + len(
        article_results["paragraph_pred"]["Methods"]
    )

    pred_results = [article_results]
    valid_num = len(test_name_seqs_df)
    logger.info(f"{total_saved_num = }, {len(para_normalized) = }")
    assert total_saved_num == valid_num, f'{total_saved_num = }, {valid_num = }'
    if save_tmp_file:
        with open(FAKE_RAW_PRED_FILE, "w", encoding="utf-8") as f:
            json.dump(pred_results, f, ensure_ascii=False, indent=4)
        logger.info(f'save tmp result file to {FAKE_RAW_PRED_FILE = }')
    return pred_results


def check(log_head_tail=0):
    ic(test_name_seqs_df.shape)
    if log_head_tail:
        ic(test_name_seqs_df.head())
        ic(test_name_seqs_df.tail())

    names_to_check = ["Metroanilide2", "phosphoprotein"]
    for name in names_to_check:
        ic(name)
        selected = test_name_seqs_df[test_name_seqs_df[NAME] == name]
        seq = selected[SEQUENCE].values[0]
        ic(seq, len(seq))
        if name == "Metroanilide2":
            real_seq = '"N"-RAA'
            ic(real_seq, len(real_seq))
            assert seq == real_seq

    ic(test_name_seqs[:5])


# check()


def auto_add_names():
    """tmp_seq_and_notes_file only has 2 columns: NAME, SEQUENCE"""
    if not tmp_seq_and_notes_file.is_file():
        return
    all_tmp_seq_and_notes = pd.read_csv(tmp_seq_and_notes_file, escapechar="\\")
    assert len(all_tmp_seq_and_notes.shape) == 2
    start_i = len(test_name_seqs_df)
    end_i = start_i + len(all_tmp_seq_and_notes)
    auto_names = [f"Pep_name_{i}" for i in range(start_i, end_i)]
    all_tmp_seq_and_notes.columns = [SEQUENCE, NOTES]
    all_tmp_seq_and_notes[NAME] = auto_names
    all_tmp_seq_and_notes = all_tmp_seq_and_notes[[NAME, SEQUENCE, NOTES]]

    existent_seqs = test_name_seqs_df[SEQUENCE].tolist()
    _df = all_tmp_seq_and_notes[~all_tmp_seq_and_notes[SEQUENCE].isin(existent_seqs)]
    if len(_df) == 0:
        logger.info("No new sequences to add. Exiting.")
        return
    logger.info(f"Write new {len(_df)} test peptide name and seq into test file")
    ic(all_tmp_seq_and_notes.shape)
    ic(existent_seqs[:3])
    new_data_to_add = _df[[NAME, SEQUENCE]]
    ic(new_data_to_add.head())
    ic(new_data_to_add.tail())

    new_df = pd.concat([test_name_seqs_df, _df], ignore_index=True)
    ic(new_df.shape)
    new_df.to_csv(TEST_FILE, index=False, sep=",")


if __name__ == "__main__":
    # auto_add_names()
    create_fake_result_file()