"""Each rule has to have a seq example in comments and test data to verify the rule!"""

import os
import re
import sys
from collections import defaultdict
from string import ascii_uppercase
from typing import Optional

from icecream import ic

sys.path.append(os.path.abspath("."))
ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120
from loguru import logger

from seq_retriever.peptide_info.fixed_peptides import fixed_pep_name_and_seqs
from seq_retriever.peptide_info.pep_data import (
    c_terminals,
    comm_terminals,
    cpp_seqs,
    disambigurated_terminals,
    gg_linker_pat,
    n_terminals,
    punc_at_terminals,
)
from seq_retriever.utils_comm.aa_util import (
    aa_3chars_to_1,
    comm_aas_upper,
    has_abnormal_upper_aa,
    is_natural_seq,
    triple_aas_capital,
    triple_aas_lower_with_homo,
    triple_aas_title,
)

# "SERPING1 gene product" in PMC7996148
extra_none_seqs = (
    "Table ",
    "no sequence",
    "sequence not",
    "sequence is not",
    "not provided",
    " gene ",
    re.compile(r'not a .+?peptide sequence'),
)
continuous_aa_and_num_in_prefix = re.compile(r"^[A-Za-z]\d+")
EMPTY = "None"
TOO_LONG = "TooLong"
UNAVAILABLE = "unavailable"
INVALID_SEQS = (EMPTY, TOO_LONG, UNAVAILABLE)
UNKNOWN_TRIPLET = "x"
pep_name_prefixes_to_filter = ('scramble', 'Scramble')
# GDFRYSDGTPV(AAA)NWY(AA)E, (A) is a mutation test to replace motif with A
# ^[A-Z]\d+[A-Z]$ GSEFMPKPHSEAGTAFIQTQQLHAAMADTFLEHMC31R
wrong_seq_pat = re.compile(
    r"\d+[−‐–-]\d+|\d+[.]\d+|\d{4}|\(A+\)|^\w\d\w\d{3}|[A-Z][−‐–-]peptide"
    r"|^[A-Z]+\d+([A-Z]*|mer)$|\((\d+\D+?|\D+?\d+)\)|[](]\d+[−‐–-]"
    r"|[A-Z][−‐–-]\d+[−‐–-][A-Z]"
)
wrong_seq_one_chat_seq = re.compile(r'\(.+\)\D')
seq_start_end_idx = re.compile(r"\d+[‐–-]\d+")
full_seq_start_end_idx = re.compile(r"^\d+[A-Z0-9)(‐–-]+\d*$")
idx_at_terminals = re.compile(r"^\d+|\d+$")
idx_at_parentheses_full_seq = re.compile(r"^[A-Z]\(\d+\)[A-Z]+[A-Z]\(\d+\)$")
idx_at_parentheses_pat = re.compile(r"\(\d+\)|\(\d+\)")
molecular_formula_pat = re.compile(r"^C\d+H\d+N\d*O\d*")
n_c_at_ends_pat = re.compile(r"^\"?N[′’]?[\"\s−‐–-]+\s*|\s*[\"\s−‐–-]+C[′’\"]?$")
n_at_end_pat = re.compile(r"^\"?N[′’]?[\"\s−‐–-]+\s*")
c_at_end_pat = re.compile(r"\s*[\"\s−‐–-]+C[′’\"]?$")
star_dot_at_ends_pat = re.compile(r"^[*.]+\s*|\s*[*.]+$")
full_seq_parentheses_at_ternimals = re.compile(r"^[\[(].+[])]$")
parentheses_at_ternimals = re.compile(r"^[\[(]|[])]$")
no_word_pat = re.compile(r"^\W+$")
refer_num_pat = re.compile(r"\s+\[\d+\]$")
aa_with_ind_pat = re.compile(r"^([A-Z]\d+)+$")
valid_peptide_seq_pat = re.compile(r"^[]\[\"0-9A-Za-z) (.αβγεδ*’′−‐–-]+$")
strict_aa_pat = re.compile(r"^[A-Za-z]+$")
strict_main_peptide_seq_pat = re.compile(r"^[A-Za-z]{3,}$")
blank_between_aas = re.compile(r'^[A-Za-z]+\s+[A-Za-z]+$')
square_brackets_at_n_terminal = re.compile(r"^\[(.+?)\]\.?")
dot_at_n_terminal = re.compile(r"^(.{1,4})\.")
square_brackets_at_c_terminal = re.compile(r"\.?\[(.+?)\]$")
dot_at_c_terminal = re.compile(r"\.(.{1,4})$")
parentheses_at_n_terminals = re.compile(r"^\((.+?)\)")
parentheses_at_c_terminals = re.compile(r"\((.+?)\)$")
double_parenthese_at_terminals = re.compile(r"^\((.+?)\)\.?(.+?)\.?\((.+?)\)$")
cyclic_prefix_pat = re.compile(
    r"^(.{1,4}[‐–-])?(cyclo|Cyclo|c|C|cycl|Cycl|cyc|Cyc)[\s‐–-]?[\[(]"
)
dup_fragment_one_char_pat = re.compile(r"\(([A-Z]+)\)([23456789])([‐–-]?[A-Z])?")
dup_fragnment_three_chars_pat = re.compile(
    r"\(([A-Za-z‐–-]{3,})\)([23456789])([‐–-]?[A-Z][a-z]{2})?"
)
one_char_with_hyphen_pat = re.compile(r"([A-Z][\s−‐–-]){2,}[A-Za-z\d]+")
l_hyphen_pat = re.compile(r"\W([Ll][−‐–-])[A-Za-z]{3}")
all_digits_pat = re.compile(r"^\d+$")
digits_at_end = re.compile(r"\d+$")
linker_pat = re.compile(r'^G+&')

similar_hyphens = ('−', '‐', '–')
HYPHEN = '-'
hyphens = (HYPHEN,) + similar_hyphens
digits = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
d_aa_prefixes = ("D", "d")
gene = ['A', 'G', 'C', 'T', 'U']


def convert_wrong_seq_to_none(pep_name: str, pep_seq: str, content: str):
    """convert clearly wrong peptide seq as None."""
    if not pep_seq:
        return EMPTY

    if wrong_seq_pat.search(pep_seq):
        logger.info(f"{pep_seq = } has wrong seq pattern")
        return EMPTY

    triple_aa_count_title = calc_title_triple_aa_num(pep_seq)
    if triple_aa_count_title >= 3:
        return pep_seq

    # "N-acetyl-seryl-aspartyl-lysyl-proline": "None",
    sub_seqs = split_by_hyphen(pep_seq)
    triple_aa_count_lower = calc_lower_triple_aa_num(sub_seqs) # type: ignore
    # if pep_seq == "S-allyl-DL-homocysteine":
    #     ic(triple_aa_count_lower)
    if triple_aa_count_lower >= 3:
        return pep_seq
    if triple_aa_count_lower > 0:
        logger.info(f'{triple_aa_count_lower = }, treat as None')
        return EMPTY
    if pep_seq.islower():
        logger.info(f"{pep_seq = } is lower case without triple AA")
        return EMPTY

    # peptide name has no space inside
    # case1, "SLIGKV-NH(2)": "SLIGKV-NH",
    # case2, "tetrapeptide PKEK": "PKEK",
    sub_words_of_name = pep_name.split()
    if len(sub_words_of_name) == 1:
        if pep_seq.isupper() and len(pep_seq) < len(pep_name) and pep_seq in pep_name:
            logger.info(
                f"Upper {pep_seq = } is part of peptide name {pep_name} which has not blanks, "
            )
            if pep_name in content:
                pep_seq = pep_name
                logger.info(f"Use {pep_name = } to as pep seq, as the name in content")

        if molecular_formula_pat.search(pep_seq):
            logger.info(f"{pep_seq = } is molecular formula")
            return EMPTY

    # LL37
    aa_upper_count = get_normal_aa_upper_count(pep_seq)
    if aa_upper_count < 3:
        logger.info(f"{pep_seq = } has less than 3 upper case AAs")
        return EMPTY

    # if len(sub_words) > 1, there are blanks in seq and we treat each word as a sub seq.
    sub_words = pep_seq.split()
    if len(sub_words) > 1:
        max_aa_upper_count = get_max_aa_upper_count_from_sub_seqs(sub_words)
        if max_aa_upper_count < 3:
            logger.info(f"{pep_seq = } has less than 3 upper case AAs in sub words")
            return EMPTY

    if is_shortname(pep_name, pep_seq):
        logger.info(f"{pep_seq = } is the short name of peptide name")
        return EMPTY

    if is_length_less_3(pep_seq, sub_seqs, triple_aa_count_lower):
        logger.info(f"{pep_seq = } is less than 3 AAs")
        return EMPTY

    # Clearly wrong seqs:
    # "TAT-ATXN1-CTM", 'X' in the seq
    # "D-BMAP18", 'B' in seq
    if len(sub_words) == 1:
        if "X" in pep_seq:
            logger.info(f"{pep_seq = } has unavailable upper X")
            return EMPTY

    if len(sub_seqs) == 1:
        if has_abnormal_upper_aa(sub_seqs[0]):
            logger.info(f"{sub_seqs[0] = } has unavailable upper AA")
            return EMPTY
    elif len(sub_seqs) == 2:
        if sub_seqs[1].isupper() and has_abnormal_upper_aa(sub_seqs[1]):
            logger.info(f"{sub_seqs[1] = } has unavailable upper AA")
            return EMPTY
        if sub_seqs[0].isupper() and has_abnormal_upper_aa(sub_seqs[0]):
            logger.info(f"{sub_seqs[0] = } has unavailable upper AA")
            return EMPTY
    # "N6-COOH-miniPEG", COOH is terminal modification, not a seq.
    elif len(sub_seqs) == 3:
        if sub_seqs[1] in comm_terminals:
            logger.info(f"{sub_seqs[1] = } is common terminal")
            return EMPTY

    if is_invalid_by_seq_length_from_pep_name(pep_name, pep_seq):
        logger.info(f"{pep_seq = } is invalid by seq length from peptide name")
        return EMPTY

    if is_too_short_cc_cyclic_peptide(pep_seq):
        return EMPTY

    if has_nested_parentheses(pep_seq):
        logger.info(f"{pep_seq = } has nested parentheses and treat as None")
        return EMPTY
    pep_seq = name_and_seq_has_same_sub_word(pep_name, pep_seq, sub_seqs)
    return pep_seq


def name_and_seq_has_same_sub_word(
    name: str, seq: str, sub_seqs: Optional[list[str]] = None
):
    """e.g.:
    SPA4 peptide,FITC-SPA4
    """
    if name == seq:
        return seq
    # case1
    if not name.endswith("peptide"):
        return seq
    if sub_seqs is None:
        sub_seqs = split_by_hyphen(seq)

    sub_names = name.split()
    if len(sub_names) == 1:
        sub_names = split_by_hyphen(name)

    for sub_name in sub_names:
        for sub_seq in sub_seqs:
            if sub_name == sub_seq:
                logger.info(f"{name = }, {seq = } has comm words and treat seq as None")
                return EMPTY
    return seq


def has_nested_parentheses(pep_seq: str):
    """has_nested_parentheses examples: CTpH(F)-(AAC-(K)-RRWQWR)4"""
    stack = []
    for c in pep_seq:
        if c == "(":
            stack.append(1)
        elif c == ")":
            if not stack:
                return False
            if len(stack) > 1:
                return True
            stack.pop()
    return False


def is_too_short_cc_cyclic_peptide(pep_seq: str):
    """Filter out the cys-cys at terminals peptide which length less than 5

    Examples: CHC
    """
    if 1 < len(pep_seq) < 5 and pep_seq.isupper():
        if pep_seq[0] == "C" and pep_seq[-1] == "C":
            logger.info(f"{pep_seq = } is too short cys-cys cyclic peptide")
            return True
    return False


def too_long_and_without_triple_aa(pep_seq: str, is_valid_triple_seq):
    if len(pep_seq) > 50:
        if is_valid_triple_seq:
            return False
        return True
    return False


def is_invalid_by_seq_length_from_pep_name(pep_name="CHGA47–66", pep_seq="CHR"):
    """examples:
    "CHGA47–66": "CHR"
    """
    if got := seq_start_end_idx.search(pep_name):
        start, end = split_by_hyphen(got.group())
        if len(pep_seq) < int(end) - int(start) + 1 and len(pep_seq) < 10:
            return True
    return False


def get_normal_aa_upper_count(pep_seq: str):
    normal_aa_upper_count = 0
    for char in pep_seq:
        if char in comm_aas_upper:
            normal_aa_upper_count += 1
    return normal_aa_upper_count


def split_by_hyphen(pep_seq: str):
    sub_seqs = pep_seq.split(hyphens[0])
    for i in range(1, len(hyphens)):
        if len(sub_seqs) > 1:
            break
        sub_seqs = pep_seq.split(hyphens[i])
    valid_sub_seqs = [seq for seq in sub_seqs if seq]
    return valid_sub_seqs


def calc_title_triple_aa_num(seq: str, debug=0):
    """including captical and all upper"""
    title_triple_aa_count = 0
    for tri_aa in triple_aas_title:
        count = seq.count(tri_aa)
        if debug and count:
            logger.info(f"{tri_aa = } {count = } in {seq}")
        title_triple_aa_count += count
    return title_triple_aa_count


def calc_capital_triple_aa_num(seq: str, debug=0):
    capital_triple_aa_count = 0
    for tri_aa in triple_aas_capital:
        count = seq.count(tri_aa)
        if debug and count:
            logger.info(f"{tri_aa = } {count = } in {seq}")
        capital_triple_aa_count += count
    return capital_triple_aa_count


def calc_lower_triple_aa_num(seq_or_sub_seqs: str):
    if isinstance(seq_or_sub_seqs, str):
        sub_seqs = split_by_hyphen(seq_or_sub_seqs)
    else:
        sub_seqs = seq_or_sub_seqs
    lower_triple_aa_count = 0
    for sub_seq in sub_seqs:
        for tri_aa_lower in triple_aas_lower_with_homo:
            if sub_seq.startswith(tri_aa_lower):
                lower_triple_aa_count += 1
                break
    return lower_triple_aa_count


def has_triplet(seq: str):
    """if has triplet inside, return lower case; not return None"""
    for triplet in aa_3chars_to_1:
        if triplet in seq:
            return aa_3chars_to_1[triplet].lower()


def is_length_less_3(
    pep_seq: str,
    sub_seqs_by_hyphen=None,
    triple_aa_count_lower=None,
):
    """Filter out the sequence which length less than 3"""
    if len(pep_seq) < 3:
        return True
    if sub_seqs_by_hyphen is None:
        sub_seqs_by_hyphen = split_by_hyphen(pep_seq)
    if triple_aa_count_lower is None:
        triple_aa_count_lower = calc_lower_triple_aa_num(sub_seqs_by_hyphen) # type: ignore

    if len(sub_seqs_by_hyphen) == 1:
        return False
    # len(sub_seqs) > 1 for the codes below!

    if triple_aa_count_lower == 2:
        logger.info(f"{triple_aa_count_lower = }, treat as length_less_3")
        return True

    # notice special case: "A-T-R-D-P-E-P-T-A-V-D-P-N"
    if len(sub_seqs_by_hyphen) == 2:
        # treat as single letter seq: "γ-EC", 'GHKK-Cu', "RW-BP100"
        max_ascii_upper_count = get_max_aa_upper_count_from_sub_seqs(
            sub_seqs_by_hyphen
        )
        if max_ascii_upper_count < 3:
            logger.info(f"{max_ascii_upper_count = }, treat as length_less_3")
            return True
    return False


def get_max_aa_upper_count_from_sub_seqs(sub_words):
    """
    case 1:
    N′- NCWPFQGVPLGFQAPP-C′ -> same
    case 2:
    S I P W N L E R I T P V R -> same
    case 3:
    F V P W F S K F [k G R I E] -> None
    """
    # case 1
    max_aa_upper_count = 0
    aa_upper_counts = []
    for sub_seq in sub_words:
        aa_upper_count = get_normal_aa_upper_count(sub_seq)
        aa_upper_counts.append(aa_upper_count)
    max_aa_upper_count = max(aa_upper_counts)

    if max_aa_upper_count == 1 and len(sub_words) > 2:
        # case 2
        if all(aa_upper_count == 1 for aa_upper_count in aa_upper_counts):
            max_aa_upper_count = len(sub_words)
    return max_aa_upper_count


def test_is_length_less_3():
    test_seqs = [
        "beta-alanyl-L-histidine",
        "N-Boc-(R)-Methionine",
        "β-Ala-His",
        "pyroGlu-Leu",
        "γ-EC",
        "WH",
        "Ala-Gln",
        "Gly-Sar",
        "Pam3Cys-Ala-Gly",
        "A-T-R-D-P-E-P-T-A-V-D-P-N",
        "β-Casofensin",  # treat as long seq, and filter in other filters as None.
    ]
    for seq in test_seqs:
        is_too_short = is_length_less_3(seq)
        ic(seq, is_too_short)


def is_shortname(pep_name: str, pep_seq: str):
    """pep seq is the shortname of pep name.

    examples:
    "chromofungin": "CHR"
    "type-C natriuretic peptide": "CNP",
    "C-atrial natriuretic factor": "CANF",
    'PR-39': 'RRRPRPPYLPRPRPPPFFPPRLPPRIPPGFPPRFPPRFP'
    '28\u2009kDa heat- and acid-stable phosphoprotein': 'SLDSDESEDEEDDYQQKRK'

    Triplet AAs to one letter AAs, we firstly switch name and seq in advance, and so this issue is solved in advance.
    "Gly-Pro-Ala": "GPA".

    CRD: an N terminal main, and a carbohydrate recognition domain
    """
    if pep_seq.isupper():
        # "chromofungin": "CHR"
        if pep_seq not in pep_name and pep_seq.lower() in pep_name:
            return True

        # Simple shortname judgement to collect the first char of each word in name.
        words_in_name = pep_name.split()
        first_chars = "".join([word[0] for word in words_in_name])
        possible_shortnames = [first_chars]

        more_words = []
        other_words = []
        for word in words_in_name:
            sub_words = split_by_hyphen(word)
            more_words.extend(sub_words)

            # "type-C natriuretic peptide": "CNP",
            if len(sub_words) > 1:
                _sub_words = []
                for _word in sub_words:
                    if _word.istitle():
                        _sub_words.append(_word)
                if not _sub_words:
                    _sub_words = sub_words[0]
                other_words.extend(_sub_words)
            else:
                other_words.append(word)

        possible_shortnames.append("".join([word[0] for word in more_words]))
        possible_shortnames.append("".join([word[0] for word in other_words]))

        shortnames_upper = [shortname.upper() for shortname in possible_shortnames]
        for shortname in shortnames_upper:
            if pep_seq in shortname:
                return True
    return False


def get_last_continuous_aa_index_from_left(aa, seq):
    for c_i, char in enumerate(seq):
        if char != aa:
            break
    if c_i == len(seq) - 1:
        c_i = len(seq)
    return c_i


def fix_continuous_aa_num_error(pep_name: str, pep_seq: str):
    """
    LLM sometime mis count the continuous_aa_num by increasing or decreasing. For example, R5 have 6-7 Rs

    fix error like on wrong continuous_aa_num, such as:
        'r5': 'RRRRRRR' -> 'RRRRR'
        'r8-aβ': 'rrrrrrrrr-gsnkgaiiglm' -> 'rrrrrrrrr-gsnkgaiiglm'
        R9/E9: RRRRRRRRREEEEEEEEE -> RRRRRRRRREEEEEEEEE
        k10: KKKKKKKKKK -> KKKKKKKKKK
        k7: KKKKK -> KKKKKKK

    But not convert: R3-NH2 to RRR-NH2, that's the orig seq at least have 2 continuous AAs.
    """
    got = continuous_aa_and_num_in_prefix.search(pep_name)
    if got:
        abbreviation_prefix = got.group()
        aa, num = abbreviation_prefix[0], abbreviation_prefix[1:]
        num = int(num)
        if num < 2:
            return pep_seq
        aas_lower = "".join([aa.lower()] * int(num))
        aas_upper = "".join([aa.upper()] * int(num))
        sub_seqs = split_by_hyphen(pep_seq)

        if pep_seq.startswith(aas_upper[:2]):
            left_index = get_last_continuous_aa_index_from_left(
                aas_upper[0], sub_seqs[0]
            )
            sub_seqs[0] = aas_upper + sub_seqs[0][left_index:]
        elif pep_seq.startswith(aas_lower[:2]):
            left_index = get_last_continuous_aa_index_from_left(
                aas_lower[0], sub_seqs[0]
            )
            sub_seqs[0] = aas_lower + sub_seqs[0][left_index:]
        pep_seq = "-".join(sub_seqs)
    return pep_seq


def convert_peptide_name_to_seq(pep_name: str, pep_seq, content):
    """Some times, LLM mis-treats peptide sequence as peptide name, convert it back.

    Examples:
        'RFMRNR-NH2': 'None' -> 'RFMRNR-NH2'
    """
    if pep_seq == EMPTY and (
        pep_name.endswith(c_terminals) or pep_name.startswith(n_terminals)
    ):
        _peptide_name = pep_name
        for terminal in c_terminals:
            if pep_name.endswith(terminal):
                _peptide_name = pep_name[: -len(terminal)]
                break
        for terminal in n_terminals:
            if _peptide_name.startswith(terminal):
                _peptide_name = _peptide_name[len(terminal) :]
                break
        if _peptide_name.isupper():
            # E.g., the seq is too long, and so convert to None
            pep_seq = convert_wrong_seq_to_none(pep_name, pep_name, content)
            logger.info(f"pep_seq is None, and treat {pep_name = } as seq {pep_seq}")
    return pep_seq


def drop_colon(pep_seq: str):
    items = pep_seq.split(": ", maxsplit=1)
    if len(items) == 2:
        logger.info(f"Drop colon from {pep_seq = }")
        pep_seq = items[1].strip()
    return pep_seq


def drop_start_end_ind_at_terminals(pep_seq: str = "123LPEHAIVQF3"):
    """Index include the end index, so the length is end-start+1"""
    if full_seq_start_end_idx.search(pep_seq):
        logger.info(f"drop_start_end_ind_at_terminals, {pep_seq = }")
        pep_seq = idx_at_terminals.sub("", pep_seq)
    return pep_seq


def drop_parenthese_at_terminal(pep_seq):
    """case 1:
    only deletes the parenthese when parentheses_count == 1
    [Gly-Gly-Gly-Gly-Ser (GGGGS)]
    """
    if full_seq_parentheses_at_ternimals.search(pep_seq):
        parentheses_count = max(pep_seq.count("("), pep_seq.count("["))
        if parentheses_count == 1:
            pep_seq = parentheses_at_ternimals.sub("", pep_seq)
    return pep_seq


def drop_explaination_in_parenthesis(
    pep_seq: str = "[Gly-Gly-Gly-Gly-Ser (GGGGS)]",
):
    """Only keep the first part before the explaination in parenthesis, e.g.:

    case 1:
    FHLGHLK (residues 19–25 of α(s1)-casein)
    CYTWNQMNL-miniPEG2-K (TYLPANASL-miniPEG2)
    Gly-Gly-Gly-Gly-Ser (GGGGS)

    case 2:
    RGCRL [36]
    """
    # case 1
    if pep_seq.endswith(")"):
        left_index = pep_seq.find("(")
        if left_index != -1 and left_index > 0:
            left_content = pep_seq[:left_index].strip()
            right_content = pep_seq[left_index + 1 : -1].strip()
            if is_shortname(left_content, right_content):
                logger.info(
                    f"content inside patentheses are shortname, and keep the left part of parenthesis {pep_seq = }"
                )
                return left_content
            lower_triple_aa_num = calc_lower_triple_aa_num(left_content)
            # γ3 peptide (NNQKIVNLKEKVAQLEA)
            if get_normal_aa_upper_count(left_content) > 2 or lower_triple_aa_num > 2:
                logger.info(f"Only keep the left part of parenthesis {pep_seq = }")
                return left_content

    # case 2
    pep_seq = refer_num_pat.sub("", pep_seq)
    return pep_seq


def drop_molecular_formula(pep_seq: str, sub_words):
    """ "C63H114N22O13S4, H–I-P-R-C-R-K-M-P–NH2" """
    if molecular_formula_pat.match(sub_words[0]):
        logger.info(f"{pep_seq = } -> {sub_words[1] = }")
        return sub_words[1]
    elif molecular_formula_pat.match(sub_words[1]):
        _pep_seq = punc_at_terminals.sub("", sub_words[0])
        logger.info(f"{pep_seq = } -> {_pep_seq = }")
        pep_seq = _pep_seq
    return pep_seq

def drop_sub_words(pep_seq: str, debug=0):
    """there are blanks or comma inside the seq, and we treat each word as a sub seq.
    Only keeps the first valid sequence.

    "C63H114N22O13S4, H–I-P-R-C-R-K-M-P–NH2",
    "AA253–279, PQQPQQFPQQQQFPQQQSPQQQQFPQQ",
    'Y-βP-WFGG-RRRRR, YaWFGG-RRRRR',

    """
    words = pep_seq.split()
    if len(words) == 2:
        pep_seq = drop_molecular_formula(pep_seq, words)

    words = pep_seq.split()
    valid_words = []
    for word in words:
        if not wrong_seq_pat.search(word) or word.endswith(comm_terminals):
            valid_words.append(word)
        else:
            logger.info(f"{word = } has wrong seq pattern, skip")
    if debug:
        ic(valid_words)

    # 'Y-βP-WFGG-RRRRR, YaWFGG-RRRRR', only select the first valid sub word as seq
    # skip "N"-RQIKIWFQNRRMKWK, KENFLRDTWCNFQFY-"C" -> RQIKIWFQNRRMKWKKENFLRDTWCNFQFY
    if n_at_end_pat.search(pep_seq) and c_at_end_pat.search(pep_seq):
        return pep_seq
    final_valid_words = []
    has_first_valid_word = False
    for word in valid_words:
        is_first_valid_seq = word.endswith(",") and (
            calc_title_triple_aa_num(word) >= 3
            or calc_lower_triple_aa_num(word) >= 3
            or get_normal_aa_upper_count(word) >= 3
        )
        if not has_first_valid_word:
            if is_first_valid_seq:
                has_first_valid_word = True
                word = word[:-1]
            final_valid_words.append(word)
        if debug:
            ic(word, is_first_valid_seq, has_first_valid_word, final_valid_words)

    _peptide_seq = " ".join(final_valid_words)
    if _peptide_seq != pep_seq:
        logger.info(f"{pep_seq = } -> {_peptide_seq = }")
    return _peptide_seq


def switch_name_and_seq(pep_name: str, pep_seq: str):
    """When name has triplet, swith peptide name and seq.

    examples1:
    "Phe-Phe-Val-Ala-Pro": "CEI5",
    "Phe-Phe-Va1-A1a-Pro-Phe-Pro-G1u-Va1-Phe-G1y-Lys": "CEI12",
    "Leu-Phe-Phe-Lys-Tyr": "None",

    examples2: PMC6966174, Analogs of SAC. SMC is not peptide seq but a name!
    "S-methyl-L-cysteine": "SMC",
    "S-allyl-D-cysteine": "ent-SAC",
    "S-propynyl-L-cysteine": "SPNC",
    "S-allyl-DL-homocysteine": "rac-SAHC",

    examples3:
    "Ac-RKLRKRLLRDWLKAFYDKVAEKLKEAF-NH2": "None"

    Not switched examples1: PMC9207436, not switch when pep_seq has modified at terminals
    "Ac-Ala-Aic-Ala-NH2": "Ac-YGKAAAAKAAAKAAAAK-NH2",
    "Histatin 5": "DSHAKRHHGYKRKFHEKHHSHRGY",
    """
    if pep_name == pep_seq:
        return pep_name, pep_seq

    if pep_seq == EMPTY and (
        pep_name.startswith(n_terminals) or pep_name.endswith(c_terminals)
    ):
        return pep_name, pep_name

    #  Not switched examples1
    if pep_seq.startswith(n_terminals) or pep_seq.endswith(c_terminals):
        return pep_name, pep_seq

    # examples1
    title_triple_aa_count_in_name = calc_title_triple_aa_num(pep_name)
    title_triple_aa_count_in_seq = calc_title_triple_aa_num(pep_seq)
    if title_triple_aa_count_in_name >= 3 and title_triple_aa_count_in_seq == 0:
        logger.info(f'orig {pep_name = }, {pep_seq = }')
        if pep_seq == EMPTY:
            pep_seq = pep_name
        pep_name, pep_seq = pep_seq, pep_name
        logger.info(f"switched as {pep_name = } and {pep_seq = }")
        return pep_name, pep_seq

    # examples2
    sub_seqs_of_name = split_by_hyphen(pep_name)
    # The pep name must have hyphens to be treated as low case sequence.
    # not switch, "Histatin 5": "DSHAKRHHGYKRKFHEKHHSHRGY",
    if len(sub_seqs_of_name) < 2:
        return pep_name, pep_seq
    lower_triple_aa_count_in_name = calc_lower_triple_aa_num(sub_seqs_of_name) # type: ignore
    if lower_triple_aa_count_in_name > 0:
        if pep_seq == EMPTY:
            # Treats name as invalid sequence and so not switch, when pep_seq is None
            if lower_triple_aa_count_in_name < 3:
                return pep_name, pep_seq
            pep_seq = pep_name
        pep_name, pep_seq = pep_seq, pep_name
        logger.warning(
            f"There are lower case triplets in peptide name, switch {pep_name = } and {pep_seq = }"
        )

    return pep_name, pep_seq


def postprocess_seq_and_name(pep_name: str, pep_seq: str, content="", excl_prefixes=()):
    """Note: after postprocess, if the sequence is not None, we can find the seq in the content!

    So, not modify the peptide sequence or name at the middle, but possibly truncate from the start or end!
    """
    # logger.debug(f'orig {pep_name = }, {pep_seq = }')
    if no_word_pat.search(pep_name):
        logger.info(f"No word in {pep_name = } and so use {pep_seq = } as name")
        pep_name = pep_seq
    pep_name, pep_seq = switch_name_and_seq(pep_name, pep_seq)
    if pep_seq in INVALID_SEQS:
        return pep_name, pep_seq
    if pep_seq.startswith(EMPTY):
        return pep_name, EMPTY
    # peptide name': 'Table 1'
    for extra_none_seq in extra_none_seqs:
        if isinstance(extra_none_seq, str):
            if extra_none_seq in pep_seq:
                logger.info(f"{extra_none_seq = } in {pep_seq = }")
                return pep_name, EMPTY
        elif isinstance(extra_none_seq, re.Pattern):
            if extra_none_seq.search(pep_seq):
                logger.info(f"extra_none_seq pattern in {pep_seq = }")
                return pep_name, EMPTY
    if excl_prefixes:
        if pep_seq.startswith(excl_prefixes):
            logger.info(f"{pep_seq = } starts with {excl_prefixes[:5] = }")
            return pep_name, EMPTY
    pep_seq = process_peptide_seq(pep_name, pep_seq, content)

    return pep_name, pep_seq


def extend_seq(pep_seq: str = "RRTQNRRNRRT", content: str = "CARP (ABRRTQNRRNRRT)"):
    """Extend the peptide seq by the content text.

    Example:
    PMC7052017
    Historical Overview of CARPs and Neuroprotection Studies
    real pep seq and name:
    a randomly designed CARP (RRRTQNRRNRRTSRQNRRRSRRRR; net charge +15)
    But model get RRTQNRRNRRTSRQNRRRSRRRR, 1R missing at the start.

    """
    seq_count = content.count(pep_seq)
    if seq_count == 1:
        for aa in pep_seq:
            # all chars must be ascii_uppercase, not number or punctuation
            if aa not in ascii_uppercase:
                return pep_seq
        seq_start_i = content.find(pep_seq)
        seq_end_i = seq_start_i + len(pep_seq)
        left_extended = []
        right_extended = []
        for i in range(seq_start_i - 1, -1, -1):
            if content[i] in ascii_uppercase:
                left_extended.append(content[i])
            else:
                break
        for i in range(seq_end_i, len(content)):
            if content[i] in ascii_uppercase:
                right_extended.append(content[i])
            else:
                break
        if left_extended or right_extended:
            left_extended.reverse()
            extended_seq = "".join(left_extended) + pep_seq + "".join(right_extended)
            logger.info(f"{pep_seq = } is extended to {extended_seq}")
            pep_seq = extended_seq
    return pep_seq


def drop_n_c_at_ends(pep_seq: str):
    """ """
     # "N"-RQIKIWFQNRRMKWK, KENFLRDTWCNFQFY-"C" -> RQIKIWFQNRRMKWKKENFLRDTWCNFQFY
    words = pep_seq.split()
    if len(words) == 2:
        if n_at_end_pat.search(words[0]) and c_at_end_pat.search(words[1]):
            logger.info(f"{pep_seq = } has N and C terminals")
            seq0 = n_at_end_pat.sub("", words[0])
            seq1 = c_at_end_pat.sub("", words[1])
            if seq0.endswith(","):
                seq0 = seq0[:-1]
            pep_seq = seq0 + seq1

    _pep_seq = n_c_at_ends_pat.sub("", pep_seq)
    if _pep_seq != pep_seq:
        logger.info(f"{pep_seq = } -> {_pep_seq = }")
        pep_seq = _pep_seq
    return pep_seq


def drop_square_brackets_at_terminals(
    pep_seq: str = "[Cy7]LRELHLNNNC", n_mod="", c_mod=""
):
    """Drop the square brackets at terminals, e.g.:
    [A].GQFPLGGVA.[A] -> GQFPLGGVA
    [Cy7]LRELHLNNNC[COOH] -> LRELHLNNNC
    (H)Gly-Gly-Phe-Leu(OMe) -> Gly-Gly-Phe-Leu
    GAGGFPGFGV.[G] -> GAGGFPGFGV
    """
    n_mod_2, c_mod_2 = "", ""
    if got_n:=square_brackets_at_n_terminal.search(pep_seq):
        n_mod = got_n.group(1)
        pep_seq = square_brackets_at_n_terminal.sub("", pep_seq)
        logger.info(f'-> {pep_seq = }')
    elif got_n_2:=dot_at_n_terminal.search(pep_seq):
        n_mod_2 = got_n_2.group(1)
        pep_seq = dot_at_n_terminal.sub("", pep_seq)
        logger.info(f'-> {pep_seq = }')

    n_mod = n_mod or n_mod_2
    if got_c:=square_brackets_at_c_terminal.search(pep_seq):
        c_mod = got_c.group(1)
        pep_seq = square_brackets_at_c_terminal.sub("", pep_seq)
        logger.info(f'-> {pep_seq = }')
    elif got_c_2:=dot_at_c_terminal.search(pep_seq):
        c_mod_2 = got_c_2.group(1)
        pep_seq = dot_at_c_terminal.sub("", pep_seq)
        logger.info(f'-> {pep_seq = }')

    c_mod = c_mod or c_mod_2
    if got_n or got_c:
        logger.info(f"-> {n_mod = }, {c_mod = }, {pep_seq = }")
    if got := double_parenthese_at_terminals.search(pep_seq):
        logger.info(f"double_parenthese_at_terminals {pep_seq = } -> {got.group(2)}")
        pep_seq = got.group(2)
        n_mod = got.group(1)
        c_mod = got.group(3)
    return pep_seq, n_mod, c_mod


def drop_cpp_from_hybrid_peptide(pep_seq):
    """cell_penetrating_peptide (cpp) is used in hybrid peptide, e.g.:

    RRRRR-GIGKFLKKAKKFGKAFVK-NH2
    KNLRIFRKGIHIHKKY-GG-KRKRWHW
    RVVQVWFQNQRAKLKK-G-KKLFKKILKKL
    MIASHLLAYFFTELN(β-Ala)-GYGRKKRRQRRRG-amide
    """
    sub_seqs = split_by_hyphen(pep_seq)
    if len(sub_seqs) == 1:
        return pep_seq
    for hyphen in hyphens:
        if hyphen in pep_seq:
            break

    new_seqs = []
    for sub_seq in sub_seqs:
        if sub_seq in cpp_seqs:
            logger.info(f"{sub_seq = } is cell penetrating peptide")
            continue
        new_seqs.append(sub_seq)
    if len(new_seqs) != len(sub_seqs):
        seqs_without_gg_linker = []
        for seq in new_seqs:
            if gg_linker_pat.search(seq):
                logger.info(f"{seq = } is GG linker")
                continue
            seqs_without_gg_linker.append(seq)
        new_seqs = seqs_without_gg_linker

    pep_seq = hyphen.join(new_seqs)
    return pep_seq


def process_peptide_seq(pep_name: str, pep_seq: str, content: str):
    """Note: not modify the peptide sequence or name at the middle, but just possibly truncate from the start or end!

    Args:
        content: content text, used to filter out the unavailable seqs.
    """
    orig_seq = pep_seq
    logger.info(f"Before postprocess_peptide seq, input {pep_name = }, {pep_seq = }")
    # Not check pep-name in content, as some peptide name are slightly changed in pred:
    # Histatin 5ΔMB (Hst 5 ΔMB) -> Histatin 5ΔMB (Hst 5ΔMB)

    ### Drops here don't break the key sequence. We need to judge the anti inflammatory
    # of the postprocessed seq within the content.
    pep_seq = drop_parenthese_at_terminal(pep_seq)
    pep_seq = drop_explaination_in_parenthesis(pep_seq)
    pep_seq = drop_colon(pep_seq)  # only for 8*8b
    pep_seq = drop_sub_words(pep_seq)

    pep_seq = convert_wrong_seq_to_none(pep_name, pep_seq, content)
    pep_seq = convert_peptide_name_to_seq(pep_name, pep_seq, content)

    if content and not pep_seq.startswith(EMPTY) and pep_seq not in content:
        logger.warning(
            f"{pep_name = } {pep_seq = } is filter out, as it is not in {content[:100] =}"
        )
        pep_seq = UNAVAILABLE

    if not (
        n_at_end_pat.search(pep_seq) and c_at_end_pat.search(pep_seq)
    ) and not valid_peptide_seq_pat.search(pep_seq):
        logger.info(f"{pep_seq = } is invalid, by valid_peptide_seq pat")
        return EMPTY
    pep_seq = extend_seq(pep_seq, content)
    logger.info(f'After postprocess_peptide seq, {orig_seq = } -> {pep_seq = }')
    return pep_seq


def parse_reply(reply: str):
    """reply example:
    Here is the list of peptide sequences extracted from the input section content:

    * copper glycine-histidine-lysine: Cu-GHK
    * tetrapeptide PKEK: PKEK
    * CPPAIF: None
    * tat: RKKRRQRRR
    * R7: None
    * R9D: Table 1
    """
    lines = reply.split("\n")
    result = defaultdict(set)
    for line in lines:
        if not line:
            continue
        line = line.strip()
        if line.startswith("* "):
            parts = line.split(":", maxsplit=1)
            if len(parts) == 2:
                pep_name = parts[0][2:]
                pep_seq = parts[1].strip()
                if not pep_name.startswith(EMPTY):
                    result[pep_name].add(pep_seq)
    return result


def is_invalid_triplet(seq: str):
    if (
        (len(seq) == 3 and (seq.isupper() or seq.istitle()))
        # seq starts with triple_aas_title, but not triple_aas_title.
        or seq.startswith(triple_aas_title)
    ):
        logger.info(f"sub {seq = } is invalid tripet")
        return True


def convert_triplet_to_one(pep_seq: str, sub_seqs: list[str], n_mod="", c_mod=""):
    """e.g.:
    case1: Leu-Phe-Phe-Lys-Tyr -> LFFKY
    case2: Ala-Ala-Pro-Val-AMC -> AAPV
    case3: Trp-Lys-Tyr-Met-Val-D-Met -> WKYMVm
    case4: (Val-Leu-Phe-Pro)2 -> VLFPVLFP
    Val-Leu-DPhe-Pro -> VLfP
    Nap-dPhe-dPhe-dLys-dTyr -> ffky

    CysLysGlyGlyArgAlaLysAspCysGlyGlyAsp -> CKGGRAKDCGGD
    """

    _pep_seq = pep_seq
    logger.info(f"{pep_seq = } is triplet seq, {sub_seqs = }")
    debug = 0
    if debug:
        for _seq in sub_seqs:
            if _seq not in aa_3chars_to_1:
                logger.info(f'{_seq = } is invalid triplet')
                break
    # case1 is the most common and simple case
    if len(sub_seqs) > 1:
        single_chars = []
        is_d_aa = False
        for i, seq in enumerate(sub_seqs):
            words = seq.split()
            # logger.info(f'{i = }, {seq}')
            if len(words) > 1:
                if len(words) == 2 and words[-1].lower() == "acid":
                    logger.info(f"{words = } is treat as {words[0]}")
                    seq = words[0]
                else:
                    logger.info(
                        f"sub {seq = } has more than 1 words, treat the whole as empty"
                    )
                    return EMPTY, n_mod, c_mod
            char = aa_3chars_to_1.get(seq, UNKNOWN_TRIPLET)
            if char == UNKNOWN_TRIPLET:
                if seq[0] in d_aa_prefixes:
                    char = aa_3chars_to_1.get(seq[1:], UNKNOWN_TRIPLET)
                    if char != UNKNOWN_TRIPLET:
                        is_d_aa = True
                elif seq[0] == 'L':
                    char = aa_3chars_to_1.get(seq[1:], UNKNOWN_TRIPLET)
                elif full_seq_parentheses_at_ternimals.findall(seq):
                    seq = seq[1:-1]
                    char = aa_3chars_to_1.get(seq, UNKNOWN_TRIPLET)
                # His6-Phe-Arg-Trp9
                elif got := digits_at_end.search(seq):
                    char = aa_3chars_to_1.get(seq[: got.start()], UNKNOWN_TRIPLET)
                else:
                    for terminal in disambigurated_terminals:
                        if seq.startswith(terminal):
                            _seq = seq[len(terminal) :]
                            char = aa_3chars_to_1.get(_seq, UNKNOWN_TRIPLET)
                            break
            if char == UNKNOWN_TRIPLET:
                # case2, n/c mods
                # logger.info(f'X {i = }, {seq}')
                if i == 0:
                    # fMet-Leu-Phe -> MLF
                    if seq.startswith("f"):
                        char = aa_3chars_to_1.get(seq[1:], UNKNOWN_TRIPLET)
                        if char != UNKNOWN_TRIPLET:
                            single_chars.append(char)
                            n_mod = "f"
                    elif got := parentheses_at_n_terminals.search(seq):
                        _seq = seq[seq.find(")") + 1 :]
                        char = aa_3chars_to_1.get(_seq, UNKNOWN_TRIPLET)
                        if char != UNKNOWN_TRIPLET:
                            single_chars.append(char)
                            n_mod = got.group(1)
                    elif seq in d_aa_prefixes:
                        is_d_aa = True
                    else:
                        n_mod = seq
                elif i == len(sub_seqs) - 1:
                    if got := parentheses_at_c_terminals.search(seq):
                        _seq = seq[: seq.find("(")]
                        char = aa_3chars_to_1.get(_seq, UNKNOWN_TRIPLET)
                        if char != UNKNOWN_TRIPLET:
                            single_chars.append(char)
                            c_mod = got.group(1)
                    else:
                        for terminal in disambigurated_terminals:
                            if seq.endswith(terminal):
                                _seq = seq[: -len(terminal)]
                                char = aa_3chars_to_1.get(_seq, UNKNOWN_TRIPLET)
                                if char != UNKNOWN_TRIPLET:
                                    single_chars.append(char)
                                    c_mod = terminal
                                    break
                        else:
                            c_mod = seq
                # case3, D aa
                elif seq in d_aa_prefixes:
                    is_d_aa = True
                elif seq == 'L':
                    continue
                # case4, for-Met-Leu-Cys(OMe)-Cys-Leu-Met-fpr
                elif seq.endswith(')'):
                    left_index = seq.find("(")
                    if left_index != -1 and left_index > 0:
                        left_content = seq[:left_index].strip()
                        char = aa_3chars_to_1.get(left_content, UNKNOWN_TRIPLET)
                        single_chars.append(char)
                elif i == 1:
                    # the second sub seq is abnormal AA
                    if is_invalid_triplet(seq):
                        return EMPTY, n_mod, c_mod
                    elif has_triplet(seq):
                        _char = has_triplet(seq)
                        logger.info(
                            f"{seq = } has substandard triplet inside, treat as {_char}"
                        )
                        single_chars.append(_char)
                    elif n_mod:
                        n_mod = f'{n_mod}-{seq}'
                    else:
                        logger.info(f"{pep_seq = } as None")
                        return EMPTY, n_mod, c_mod
                elif i == len(sub_seqs) - 2:
                    # the last second sub seq is abnormal AA
                    if is_invalid_triplet(seq):
                        return EMPTY, n_mod, c_mod
                    else:
                        c_mod = seq
                else:
                    return EMPTY, n_mod, c_mod
            else:
                if is_d_aa:
                    char = char.lower()
                    is_d_aa = False
                single_chars.append(char)

        _pep_seq = "".join(single_chars)
    # CysLysGlyGlyArgAlaLysAspCysGlyGlyAsp ->  CKGGRAKDCGGD
    else:
        single_chars = []
        for start_i in range(0, len(pep_seq), 3):
            seq = pep_seq[start_i : start_i + 3]
            char = aa_3chars_to_1.get(seq, UNKNOWN_TRIPLET)
            if char == UNKNOWN_TRIPLET:
                return EMPTY, n_mod, c_mod
            single_chars.append(char)
        _pep_seq = "".join(single_chars)
    logger.info(f"{pep_seq} -> {_pep_seq = }, {n_mod = } {c_mod = }")
    pep_seq = _pep_seq
    return pep_seq, n_mod, c_mod


def convert_aa_with_each_ind(seq: str):
    """ e.g.: 
    V1P2T3L4P5L6V7P8L9G10 -> VPTLPLVPLG, 
    R0Q1L2V3L4G5L6 -> QLVLGL, R0 is the n-terminal modification
    """
    aas = []
    indexes = []
    ind = ''
    for char in seq:
        if char in ascii_uppercase:
            aas.append(char)
            if ind:
                indexes.append(int(ind))
            ind = ''
        elif char in digits:
            ind += char
    if ind:
        indexes.append(int(ind))
    if indexes and (len(aas) == len(indexes)) and (indexes[0] == 0 or indexes[0] == 1):
        valid_aas = []
        for aa, ind in zip(aas, indexes):
            if ind == 0:
                continue
            valid_aas.append(aa)
        logger.info(f'{seq = } has index for each AA')
        seq = ''.join(valid_aas)
    return seq


def convert_aa_with_ind_at_terminal(seq: str):
    """ e.g.: 
    M(1)IASHLLAYFFTELA(15) -> MIASHLLAYFFTELA
    """
    if idx_at_parentheses_full_seq.search(seq):
        logger.info(f'{seq = } has index for terminal aas.')
        seq = idx_at_parentheses_pat.sub('', seq)
    return seq


def convert_one_char_seq(pep_seq: str, sub_seqs: list[str], n_mod="", c_mod=""):
    """e.g.:
    
    Npx-DFDFGSSSR -> DFDFGSSSR,Npx
    Ac-YGKAAAAKAAAKAAAAK-NH2
    Stearyl–HSDAVFTDNYTRLRKQ-Nle-AVKKYLNSILN-NH2 -> None

    'V-S-P-Y-K-S-G-M-W-E-R-H-F' -> VSPYKSGMWERHF
    V1P2T3L4P5L6V7P8L9G10 -> VPTLPLVPLG
    M(1)IASHLLAYFFTELA(15) -> MIASHLLAYFFTELA
    """
    logger.info(f"{pep_seq = } is one_char seq, {sub_seqs = }")
    orig_seq = pep_seq
    if len(sub_seqs) == 1:
        pep_seq, n_mod, c_mod = normalize_main_seq(pep_seq)
        logger.info(f'After first normalize main_seq {pep_seq = }')
        # from table
        # Clavanin A (VFQFLGKIIHHVGNFVHGFSHVF-NH2)
        # γ3 peptide (NNQKIVNLKEKVAQLEA)
        sub_words = pep_seq.split()
        if len(sub_words) > 1 and sub_words[-1].endswith(')') and sub_words[-1].startswith('('):
            pep_seq = sub_words[-1][1:-1]
            pep_seq, n_mod, c_mod = normalize_main_seq(pep_seq)
        elif len(sub_words) == 1 and sub_words[0].endswith(']'):
            pep_seq = sub_words[0][0:-1]

    # case2 Npx-DFDFGSSSR -> DFDFGSSSR,Npx
    elif len(sub_seqs) == 2:
        if not sub_seqs[0].isupper():
            if sub_seqs[1] in comm_terminals:
                pep_seq = sub_seqs[0]
                c_mod = sub_seqs[1]
            else:
                n_mod = sub_seqs[0]
                pep_seq = sub_seqs[1]
        elif len(sub_seqs[0]) < 3 and len(sub_seqs[1]) > 3:
            pep_seq = sub_seqs[1]
            c_mod = sub_seqs[0]
        elif pep_seq.isupper() and sub_seqs[1] not in comm_terminals:
            logger.info(f'{pep_seq = } are hybrid seq, treat as empty')
            pep_seq = EMPTY
        elif sub_seqs[0] in comm_terminals:
            n_mod = sub_seqs[0]
            for term in disambigurated_terminals:
                if sub_seqs[1].endswith(term):
                    pep_seq = sub_seqs[1][: -len(term)]
                    c_mod = term
                    break
            else:
                pep_seq = sub_seqs[1]
        else:
            pep_seq = sub_seqs[0]
            c_mod = sub_seqs[1]
    # case3
    elif len(sub_seqs) == 3:
        seq0 = sub_seqs[0].strip()
        seq1 = sub_seqs[1].strip()
        seq2 = sub_seqs[2].strip()
        if blank_between_aas.search(seq1):
            seq1 = seq1.replace(' ', '')
        seq_2_valid = strict_main_peptide_seq_pat.search(seq2)
        seq_1_valid = strict_main_peptide_seq_pat.search(seq1)
        seq_0_valid = strict_main_peptide_seq_pat.search(seq0)
        if seq1.isupper() and seq_1_valid and seq_0_valid and seq0.isupper():
            logger.info(
                f"{sub_seqs[0] = } and {sub_seqs[1] = } are both valid seqs, ambigurous, and treat the seq as empty"
            )
        elif seq_2_valid and seq_1_valid and seq_0_valid:
            logger.info(
                f"{sub_seqs[0] = }, {sub_seqs[1] = } and {sub_seqs[2] = } are all valid seqs, ambigurous, and treat the seq as empty"
            )
            pep_seq = EMPTY
        elif seq_0_valid and seq0 not in comm_terminals:
            pep_seq = seq0
            if not linker_pat.search(seq1):
                c_mod = f"{seq1}-{seq2}"
            else:
                logger.info(f'{seq1 = } are linkers and treat the whole seq as empty')
            pep_seq = EMPTY
        # if seq1 is main seq and the seq2 following should not be all digits, e.g.:
        # C-FhHDM-1 is not a pep seq
        elif seq_1_valid and not all_digits_pat.search(seq2):
            pep_seq = seq1
            n_mod = seq0
            c_mod = seq2
        else:
            logger.info(f'No valid sub seqs in {sub_seqs = }')
            pep_seq = EMPTY
            n_mod = seq0
            c_mod = seq2
    # H–I-P-R-C-R-K-M-P-G-V-K-M-C–NH2, the first and last hyphens diff with that in mid
    elif got := one_char_with_hyphen_pat.search(pep_seq):
        seq_with_hyphen = got.group()
        chars = split_by_hyphen(seq_with_hyphen)
        if len(chars) == 1:
            chars = seq_with_hyphen.split()
        has_hyphen = HYPHEN in pep_seq
        logger.info(f"{pep_seq = }, {seq_with_hyphen = }, {chars = } {has_hyphen = }")
        pep_seq = "".join(chars)
        if has_hyphen:
            for hyphen in similar_hyphens:
                if hyphen in pep_seq:
                    logger.info(f'{hyphen = } and {HYPHEN} both in {pep_seq = }')
                    items = pep_seq.split(hyphen)
                    if len(items) == 1:
                        pass
                    elif len(items) < 4:
                        pep_seq = items[1]
                    else:
                        logger.info(f'{pep_seq = } has more than 3 hyphens {hyphen}, treat as empty')
                        pep_seq = EMPTY
                    break
    else:
        # case4, Larger than 3 parts and cannot judge which is main seq, treat as empty
        new_sub_seqs = []
        for sub_seq_item in sub_seqs:
            if (
                sub_seq_item not in comm_terminals
                and len(sub_seq_item) > 3
            ):
                new_sub_seqs.append(sub_seq_item)
        logger.info(f"{new_sub_seqs = }")
        if len(new_sub_seqs) == 1:
            pep_seq = new_sub_seqs[0]
        ## stricter rules to keep high precision and so comments the elif
        # elif len(new_sub_seqs) == 2:
        #     if (
        #         not new_sub_seqs[0].isupper()
        #         and len(new_sub_seqs[0]) < 5
        #         and new_sub_seqs[1].isupper()
        #     ):
        #         pep_seq = new_sub_seqs[1]
        #         n_mod = new_sub_seqs[0]
        else:
            logger.warning(
                f"{pep_seq = } has more than 3 sub seqs, and cannots judge which is main seq, treat as empty"
            )
            pep_seq = EMPTY

    # MCKYFIKIVSKSAKK(FITC)PVGLIGC -> None
    if not strict_main_peptide_seq_pat.search(pep_seq):
        logger.info(f"{pep_seq = } is not strict_main_peptide_seq, treat as empty")
        pep_seq = EMPTY

    if pep_seq != orig_seq:
        logger.info(f"{pep_seq = } {n_mod = } {c_mod = }")
    return pep_seq, n_mod, c_mod


def normalize_main_seq(pep_seq):
    pep_seq = convert_aa_with_ind_at_terminal(pep_seq)
    if wrong_seq_one_chat_seq.search(pep_seq):
        logger.info(f'{pep_seq = } convert to None as it has wrong_seq_one_chat_seq')
        pep_seq = EMPTY
    pep_seq = convert_aa_with_each_ind(pep_seq)
    # Ac MLGMIKNSLFFGSVETWPWQVLNH2
    if blank_between_aas.search(pep_seq):
        pep_seq = pep_seq.replace(' ', '')
    for term in disambigurated_terminals:
        if pep_seq.startswith(term):
            logger.info(f"del {term = } which is N terminals")
            pep_seq = pep_seq[len(term):]
    for term in disambigurated_terminals:
        if pep_seq.endswith(term):
            logger.info(f"del {term = } which is C terminals")
            pep_seq = pep_seq[: -len(term)]
    pep_seq = pep_seq.strip()

    # S I P W N L E R I T P V A -> SIPWNLERITPVA
    sub_words = pep_seq.split()
    if len(sub_words) > 2 and all(len(word) == 1 for word in sub_words):
        logger.info(f'{sub_words = }')
        pep_seq = ''.join(sub_words)

    n_mod = ""
    c_mod = ""
    # fMNPLAQ -> f, MNPLAQ, f is n-mod
    if pep_seq.startswith("f"):
        n_mod = "formyl"
        pep_seq = pep_seq[1:]
    return pep_seq, n_mod, c_mod


def drop_cyclic_prefix(pep_seq: str):
    """
    If it is cyclic peptide, not n and c ternimal, no terminal modifications.

    Drop the cyclic prefix, e.g.:
    cyclo(Val-Leu-Phe-Pro)2 -> (Val-Leu-Phe-Pro)2
    c (RGDyK) -> RGDyK
    H-cycl (Cys-Ser-Ser-Ala-Gly-Ser-Leu-Phe-Cys)-OH -> Cys-Ser-Ser-Ala-Gly-Ser-Leu-Phe-Cys-OH
    Cyclo-(IR)2P(IR)2P -> (IR)2P(IR)2P
    """
    cyclic = ''
    if cyclic_prefix_pat.search(pep_seq):
        seq = cyclic_prefix_pat.sub("", pep_seq)
        right_parentheses_idx = seq.find(")")
        if right_parentheses_idx + 1 < len(seq):
            next_char = seq[right_parentheses_idx + 1]
            # If the next char of right parentheses is not a digit, drop the right parentheses.
            if next_char not in digits:
                pep_seq = seq.replace(")", "", 1)
            else:
                pep_seq = "(" + seq
        elif right_parentheses_idx + 1 == len(seq):
            pep_seq = seq[:-1]
        logger.info(f"Drop cyclic prefix from {pep_seq = }")
        cyclic = 'cyclic'
    return pep_seq, cyclic


def convert_duplicate_fragments_to_full_seq(pep_seq):
    """
    (Val-Leu-Phe-Pro)2 -> Val-Leu-Phe-Pro-Val-Leu-Phe-Pro
    (Val-Leu-Phe-Pro)2-Lys -> Val-Leu-Phe-Pro-Val-Leu-Phe-Pro-Lys
    (IR)2P(IR)2P -> IRIRPIRIRP
    CysLysGlyGlyArgAlaLysAspCysGlyGlyAsp(LysLeuAlaLysLeuAlaLys)2 -> CKGGRAKDCGGDKLAKLAKKLAKLAK
    """
    got3 = dup_fragnment_three_chars_pat.search(pep_seq)
    got1 = dup_fragment_one_char_pat.search(pep_seq)
    while got1 or got3:
        got = got3
        if got1:
            got = got1
        fragment = got.group(1)  # type: ignore
        num = int(got.group(2))  # type: ignore
        hyphen = ""
        for _hyphen in hyphens:
            if _hyphen in fragment:
                hyphen = _hyphen
                break
        full_frags = hyphen.join([fragment] * num)
        group3 = got.group(3)  # type: ignore
        if group3 is None:
            group3 = ""
        left_part_in_seq = pep_seq[: got.start()]  # type: ignore
        right_part_in_seq = pep_seq[got.end() :]  # type: ignore
        logger.info(f"convert_duplicate_fragments in {pep_seq = }")
        pep_seq = f"{left_part_in_seq}{full_frags}{group3}{right_part_in_seq}"
        got3 = dup_fragnment_three_chars_pat.search(pep_seq)
        got1 = dup_fragment_one_char_pat.search(pep_seq)
    return pep_seq


def drop_l_char(seq):
    """Gly-(L-His)-(L-Lys) -> Gly-(His)-(Lys)"""
    replaced_indexes = []
    for item in l_hyphen_pat.finditer(seq):
        replaced_indexes.append([item.start(1), item.end(1)])
    if replaced_indexes:
        _seq = ""
        start_i = 0
        for start, end in replaced_indexes:
            _seq += seq[start_i:start]
            start_i = end
        _seq += seq[start_i:]
        logger.info(f"drop L char from {seq = } -> {_seq = }")
        seq = _seq
    return seq


def filter_gene_seq(name, seq):
    """
    DNA : AGCT
    RNA : AGCU
    ABP5 shRNA,TAACCAAAGGAATGATCCT → None
    siNPRA,GGGCGCUGCUGCUGCUACCdTdT → None         dT is extra inserted，because RNA dont have T
    """
    gene_num = sum(seq.count(item) for item in gene)
    if re.findall(r'RNA|NPRA', name) and gene_num > 0.9 * len(seq):
        logger.info(f"{seq = } is RNA or DNA sequence, treat it as empety")
        seq = EMPTY
    return seq


def normalize_peptide_seq(merged_orig: dict[str, str], filter_length=0):
    """The pep name is not changed in the returned normalized_results.

    notice the name, such as "cyclic" maybe in the name of peptide"""
    normalized_results = defaultdict(list)
    for pep_name, pep_seq in merged_orig.items():
        is_natural = False
        n_mod, c_mod, cyclic = "", "", ""
        if pep_name in fixed_pep_name_and_seqs and pep_seq != EMPTY:
            pep_seq, n_mod, c_mod, cyclic, is_natural = fixed_pep_name_and_seqs[
                pep_name
            ]
            normalized_results[pep_name] = [pep_seq, n_mod, c_mod, cyclic, is_natural]
            continue
        if pep_seq in INVALID_SEQS:
            normalized_results[pep_name] = [pep_seq, n_mod, c_mod, cyclic, is_natural]
            continue
        if pep_name.startswith(pep_name_prefixes_to_filter):
            logger.info(
                f"{pep_name = } start with {pep_name_prefixes_to_filter = }, treat as None"
            )
            normalized_results[pep_name] = [EMPTY, n_mod, c_mod, cyclic, is_natural]
            continue
        pep_seq = star_dot_at_ends_pat.sub('', pep_seq)
        pep_seq, cyclic = drop_cyclic_prefix(pep_seq)
        pep_seq = drop_parenthese_at_terminal(pep_seq)
        pep_seq = convert_duplicate_fragments_to_full_seq(pep_seq)
        pep_seq = drop_start_end_ind_at_terminals(pep_seq)
        pep_seq = drop_n_c_at_ends(pep_seq)
        pep_seq, n_mod, c_mod = drop_square_brackets_at_terminals(pep_seq, n_mod, c_mod)
        pep_seq = fix_continuous_aa_num_error(pep_name, pep_seq)

        pep_seq = drop_l_char(pep_seq)
        pep_seq = filter_gene_seq(pep_name, pep_seq)
        triple_aas_title_count = calc_title_triple_aa_num(pep_seq)
        triple_aas_capital_count = calc_capital_triple_aa_num(pep_seq)
        triple_aas_lower_count = calc_lower_triple_aa_num(pep_seq)
        sub_seqs = split_by_hyphen(pep_seq)
        # Pam3Cys-SSNKSTTGSGETTTA, and so triplet nunm >= 2
        is_valid_triple_seq = (
            triple_aas_title_count > 2
            or triple_aas_lower_count >= 2
            or triple_aas_capital_count >= 2
        )
        if is_valid_triple_seq:
            pep_seq, n_mod, c_mod = convert_triplet_to_one(
                pep_seq, sub_seqs, n_mod, c_mod
            )
        else:
            pep_seq, n_mod, c_mod = convert_one_char_seq(
                pep_seq, sub_seqs, n_mod, c_mod
            )

        ## for high accuray, excludes hybird peptides, as the cell penetrating peptide is not the key sequence.
        # pep_seq = drop_cpp_from_hybrid_peptide(pep_seq)

        if too_long_and_without_triple_aa(pep_seq, is_valid_triple_seq):
            pep_seq = TOO_LONG

        # when final pep_seq endswith('peptide'), it is mostly likely a peptide name
        if pep_seq.endswith("peptide"):
            pep_seq = EMPTY

        # leaves to manually filter out to check out
        if filter_length and len(pep_seq) < 3:
            logger.info(f"{pep_name = } {pep_seq = }, seq length < 3 and treat as None")
            pep_seq = EMPTY
        if pep_seq in INVALID_SEQS:
            is_natural = False
        else:
            is_natural = is_natural_seq(pep_seq)

        normalized_result = [pep_seq, n_mod, c_mod, cyclic, is_natural]
        normalized_results[pep_name] = normalized_result
    return normalized_results


def test_fix_continuous_aa_num_error():
    """
    'r5': 'RRRRRRR' -> 'RRRRR'
    'r8-aβ': 'rrrrrrrrr-gsnkgaiiglm' -> 'rrrrrrrrr-gsnkgaiiglm'
    R9/E9: RRRRRRRRREEEEEEEEE -> RRRRRRRRREEEEEEEEE
    """
    wrong_examples = [
        ("r5", "RRRRRRR"),
        ("r8-aβ", "rrrrrrrrr-gsnkgaiiglm"),
        ("r9-socs1-kir", "RRRRRRRRRR-DTHFRTFRSHSDYRRI"),
        ("R9/E9", "RRRRRRRRREEEEEEEEE"),
        ("k10", "KKKKKKKKKK"),
        ("k7", "KKKKK"),
        ("R3-NH2", "R3-NH2"),
    ]
    for item in wrong_examples:
        print(fix_continuous_aa_num_error(*item))


def test_convert_duplicate_fragments_to_full_seq():
    seqs = [
        ("(Val-Leu-Phe-Pro)2", "Val-Leu-Phe-Pro-Val-Leu-Phe-Pro"),
        ("(Val-Leu-Phe-Pro)2-Lys", "Val-Leu-Phe-Pro-Val-Leu-Phe-Pro-Lys"),
        ("(IR)2P(IR)2P", "IRIRPIRIRP"),
        ("Asp(LysLeuAla)2", "AspLysLeuAlaLysLeuAla"),
    ]
    for seq, target_seq in seqs:
        result = convert_duplicate_fragments_to_full_seq(seq)
        assert result == target_seq, ic(seq, result, target_seq)
        # break


if __name__ == "__main__":
    ## key func: postprocess_peptide_seq, merge_sections_pred, normalize_peptide_seq
    # test_postprocess_peptide_seq()

    # test_fix_continuous_aa_num_error()
    # test_peptide_seq_pattern(test_valid_seq=1, full_search_result=1)
    # test_is_length_less_3()
    # print(drop_explaination_in_parenthesis())
    # is_invalid_by_seq_length_from_pep_name()
    # print(drop_start_end_ind_at_terminals())
    # print(drop_molecular_formula())
    # extend_seq()
    # test_convert_duplicate_fragments_to_full_seq()
    drop_square_brackets_at_terminals()