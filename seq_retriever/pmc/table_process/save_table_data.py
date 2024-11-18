import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from icecream import ic

sys.path.append(os.path.abspath('.'))
from seq_retriever.pmc.data_process import (
    orig_pmc_file,
    pmc_input_file,
    raw_table_data_file,
)

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120
SEED = 0
random.seed(SEED)
np.random.seed(SEED)


def save_table_data_file(pmc_with_table_file, table_data_file, pmc_input_file=None):
    """  
    input data is a json file with the following structure:
        list[dict] where each dict is an artile data, has the following keys, that's main_sections: ['pmc', 'pmid', 'paragraph', 'caption', 'tables', 'references']
    
    """
    pmc_ids_table = []
    pmc_with_table_data = json.load(open(pmc_with_table_file))
    ic(len(pmc_with_table_data))

    main_sections = set()
    all_table_data = []
    log_first_table_data = False
    for i, article_data in enumerate(pmc_with_table_data):
        pmc_id = article_data['pmc']
        pmc_ids_table.append(pmc_id)
        for key in article_data.keys():
            main_sections.add(key)

        table_data = article_data['tables']
        if table_data:
            _article_data = {
                'pmc': pmc_id,
                'pmid': article_data['pmid'],
                'tables': table_data,
            }
            if not log_first_table_data:
                log_first_table_data = True
                ic(table_data)
            all_table_data.append(_article_data)

    ic(len(all_table_data))
    if all_table_data:
        table_data_file = Path(table_data_file)
        first_table_data_file = table_data_file.with_stem(table_data_file.stem + '_first')
        with open(first_table_data_file, 'w', encoding='utf-8') as f:
            json.dump(all_table_data[0], f, ensure_ascii=False, indent=4)
        with open(table_data_file, 'w', encoding='utf-8') as f:
            json.dump(all_table_data, f, ensure_ascii=False, indent=4)

    pmc_ids_table = set(pmc_ids_table)
    # assert len(pmc_ids_full_text) == len(pmc_ids_table)
    ic(len(pmc_ids_table))
    ic(main_sections)

    if pmc_input_file is None:
        return

    # check ids difference when there are different version files.
    pmc_ids_full_text = []
    pmc_input_data = json.load(open(pmc_input_file))
    for article_data in pmc_input_data:
        pmc_ids_full_text.append(article_data['pmc'])
    ic(len(pmc_ids_full_text))
    pmc_ids_full_text = set(pmc_ids_full_text)

    pmc_ids_common = pmc_ids_full_text.intersection(pmc_ids_table)
    ic(len(pmc_ids_common))
    pmc_ids_only_table = pmc_ids_table.difference(pmc_ids_full_text)
    ic(len(pmc_ids_only_table), pmc_ids_only_table)
    pmc_ids_only_full_text = pmc_ids_full_text.difference(pmc_ids_table)
    ic(len(pmc_ids_only_full_text), pmc_ids_only_full_text)


if __name__ == '__main__':
    save_table_data_file(orig_pmc_file, raw_table_data_file, pmc_input_file)