from string import ascii_uppercase

from icecream import ic

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120

natural_amino_acids = (
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "L",
    "M",
    "N",
    "P",
    "K",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
)
aa_full_names = {
    "alanine": "A",
    "arginine": "R",
    "asparagine": "N",
    "aspartate": "D",
    "aspartyl": "D",
    "aspartic": "D",
    "cysteine": "C",
    "glutamine": "Q",
    "glutamate": "E",
    'glutamic': 'E',  # 'glutamic acid' is the full name of E
    "glycine": "G",
    "histidine": "H",
    "isoleucine": "I",
    "leucine": "L",
    "leucinal": "X",  # different from leucine and treat as X
    # 'norleucine': 'X' # different from leucine and treat as X
    "lysine": "K",
    "methionine": "M",
    # a wrong writing, treat as M, e.g.: N-formyl-methyl-leucyl-phenylalanine
    "methyl": "M",
    "phenylalanine": "F",
    "proline": "P",
    "serine": "S",
    "threonine": "T",
    "tryptophan": "W",
    "tyrosine": "Y",
    "valine": "V",
    "selenocysteine": "U",
    "pyrrolysine": "O",
}
aa_full_aliases = {aa.replace('ine', 'yl'): v for aa, v in aa_full_names.items()}
aa_full_names.update(aa_full_aliases)
_aa_full_names = {aa.capitalize(): v for aa, v in aa_full_names.items()}

aa_full_names.update(_aa_full_names)
_triple_aas_capital = (
    "Ala",
    "Cys",
    "Asp",
    "Glu",
    "Phe",
    "Gly",
    "His",
    "Ile",
    "Leu",
    "Met",
    "Asn",
    "Pro",
    "Lys",
    "Gln",
    "Arg",
    "Ser",
    "Thr",
    "Val",
    "Trp",
    "Tyr",
    "Pyl",
    "Sec",
)
_triple_aas_upper = tuple(aa.upper() for aa in _triple_aas_capital)
aa_3chars_to_1 = {
    "Ala": "A",
    "Cys": "C",
    "Asp": "D",
    "Glu": "E",
    "Phe": "F",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Nle": "l",  # an isomer of the more common amino acid leucine.
    'Dmt': 'X',
    "Met": "M",
    "Asn": "N",
    "Pro": "P",
    "Lys": "K",
    "Gln": "Q",
    "Arg": "R",
    "Ser": "S",
    "Thr": "T",
    "Val": "V",
    "Trp": "W",
    "Tyr": "Y",
    "Pyl": "O",
    "Sec": "U",
}
triple_aas_title_wrong_typed_dict = {
    aa.replace("l", "1"): aa for aa in _triple_aas_capital
}
triple_aas_title_wrong_typed = tuple(triple_aas_title_wrong_typed_dict.keys())
triple_aas_capital = tuple(set(_triple_aas_capital + triple_aas_title_wrong_typed))
triple_aas_upper = tuple(aa.upper() for aa in triple_aas_capital)
triple_aas_title = triple_aas_capital + triple_aas_upper
# loose logic to only exclude X and Z, include O, U, B and J
UNAVAIABLE_UPPER_AAS = ("X", "Z")
comm_aas_upper = tuple(aa for aa in ascii_uppercase if aa not in UNAVAIABLE_UPPER_AAS)
triple_aas_lower = tuple(aa.lower() for aa in triple_aas_capital)
# homo is also Unusual & Non-natural Amino Acids
triple_aas_lower_with_homo = triple_aas_lower + tuple(
    f"homo{aa}" for aa in triple_aas_lower
)

for k in triple_aas_title_wrong_typed:
    aa_3chars_to_1[k] = aa_3chars_to_1[k.replace("1", "l")]
for k in _triple_aas_upper:
    aa_3chars_to_1[k] = aa_3chars_to_1[k.capitalize()]
for k in triple_aas_lower:
    aa_3chars_to_1[k] = aa_3chars_to_1[k.capitalize()]
aa_3chars_to_1.update(aa_full_names)

def has_abnormal_upper_aa(pep_seq: str):
    for unavaiable_upper_aa in UNAVAIABLE_UPPER_AAS:
        if unavaiable_upper_aa in pep_seq:
            return True
    return False


def is_natural_seq(seq):
    for char in seq:
        if char not in natural_amino_acids:
            return False
    return True