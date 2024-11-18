import json
import math
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from icecream import ic
from loguru import logger
from pandas import DataFrame
from tqdm import tqdm

sys.path.append(os.path.abspath("."))


ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120

punc_at_terminals = re.compile(r"\W$|^\W")

# uncomm: "Suc(oMe)-AAPV-AMC", "Ac-DEVD-AMC", "suc-FLF-sBzl",
# fMNPLAQ, Formylated peptides corresponding to the N-termini

c_terminals = (
    "-COOH",
    "-CONH2",
    "-NH2",
    "-NH",
    "-NH(2)",
    "-Cu",
    "-Oxygen",
    "-OH",
    "-amide",
    "-OMe",
    '-pNA',
)
n_terminals = (
    "formyl-",
    "acetyl-",
    "Pam3-",
    "Palmitoyl-",
    "H2N-",
    "H-",
    "D-",
    "cyclo",
    "NH2-",
    "Ac-",
    "Cu-",
    "PhAc–",
    "Nap-",
    "Stearyl-",
    "Succinyl-",
    "rhodamine B-",
    "HRP-",
    "αG’-",
    "αG*-",
    'Fmoc-',
    "Z-",
    'C-'
    "Ts-",
)
# not use set to keep the order
merged_terminals = ("PEG",) + c_terminals + n_terminals
terms = []
for mod in merged_terminals:
    _mod = punc_at_terminals.sub("", mod)
    if _mod not in terms:
        terms.append(_mod)
comm_terminals = tuple(terms)
ambigurous_terms = ('NH', 'HRP', 'PEG')
disambigurated_terminals = tuple(
    term for term in comm_terminals if (len(term) > 1 and term not in ambigurous_terms)
)
cpp_seqs = pd.read_csv("seq_retriever/peptide_info/cpp.csv")["Sequence"].tolist()
gg_linker_pat = re.compile(r"^G+$")


def get_terminal_modifications():
    pass


if __name__ == "__main__":
    get_terminal_modifications()