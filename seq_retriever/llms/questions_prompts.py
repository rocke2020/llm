from loguru import logger

basic_examples_question = """
Peptide sequences are defined as sequences of amino acids and mostly have two types: one uppercase letter (A-Z) for one amino acid, for examples: RKKRRQRRR, ; and three letters (Ala, Cys, Asp, Glu, Phe, Gly, His, Ile, Leu, Met, Asn, Pro, Lys, Gln, Arg, Ser, Thr, Val, Trp, Tyr) for one amino acid and concatenated by hyphen "-", for examples: Gln-Cys-Gln-Gln-Ala-Val-Gln-Ser-Ala-Val. Peptide sequences may have extra modification or molecular letters at the start or end of sequeences, for examples: H-Ile-Pro-Arg-Cys-Arg-Lys-Met-Pro-Gly-Val-Lys-Met-Cys-NH2, Pam3Cys-Ala-Gly, Cu-GHK. Peptide sequences have the amino acids number between 3 and 50, and mostly are short sequences.
Identify peptide sequences only from the context information and not prior knowledge. If the sequence is not available in the context, just return "Unavailable", not create a fake sequence.
Output each peptide data with both its name and sequence on a separate line.
"""
more_examples_question = """
Peptide sequences are defined as sequences of amino acids and mostly have two types: one uppercase letter (A-Z) for one amino acid, for examples: FGKVREQIPP, YKKHRQRCW, INLKALAALAKKIL, etc; and three letters (Ala, Cys, Asp, Glu, Phe, Gly, His, Ile, Leu, Met, Asn, Pro, Lys, Gln, Arg, Ser, Thr, Val, Trp, Tyr) for one amino acid and concatenated by hyphen "-", for examples: Gln-Cys-Gln-Gln-Ala-Val-Gln-Ser-Ala-Val. Peptide sequences may have extra modification or molecular letters at the start or end of sequeences, for examples: H-Ile-Pro-Arg-Cys-Arg-Lys-Met-Pro-Gly-Val-Lys-Met-Cys-NH2, Pam3Cys-Ala-Gly, and Cu-GHK, etc. Peptide sequences have the amino acids number between 3 and 50, and mostly are short sequences.
Find peptide sequences only from the context information and not prior knowledge. If the sequence is not available in the context above, do not make up a sequence, but directly report unavailable.
Output each peptide data with both its name and sequence on a separate line.
"""
no_examples_question = """
Peptide sequences are defined as sequences of amino acids and mostly have two types: one uppercase letter (A-Z) for one amino acid; and three letters (Ala, Cys, Asp, Glu, Phe, Gly, His, Ile, Leu, Met, Asn, Pro, Lys, Gln, Arg, Ser, Thr, Val, Trp, Tyr) for one amino acid and concatenated by hyphen "-". Peptide sequences may have extra modification or molecular letters at the start or end of sequeences. Peptide sequences have the amino acids number between 3 and 50, and mostly are short sequences.
Identify peptide sequences only from the context information and not prior knowledge. If the sequence is not available in the context, do not generate a sequence, but treat as unavailable. Output each peptide data with both its name and sequence on a separate line.
"""
concise_question = """
Peptide sequences are defined as sequences of amino acids with the amino acids number between 3 and 50, and mostly are short sequences.
Identify peptide sequences only from the context information and not prior knowledge. If the sequence is not available in the context, just return "Unavailable", not create a fake sequence.
Output each peptide data with both its name and sequence on a separate line.
"""
shortest_question = """
Identify peptide sequences only from the context information and not prior knowledge. If the sequence is not available in the context, just return "Unavailable", not create a fake sequence.
Output each peptide data with both its name and sequence on a separate line.
"""

FULL_EXAMPLES_QUESTION = """
Peptide sequences are sequences of amino acids and mostly have two types: one uppercase letter (A-Z) for one amino acid, for examples: FGKVREQIPP, YKKHRQRCW, INLKALAALAKKIL; and three letters for one amino acid (Ala, Cys, Asp, Glu, Phe, Gly, His, Ile, Leu, Met, Asn, Pro, Lys, Gln, Arg, Ser, Thr, Val, Trp, Tyr) which are concatenated by hyphen "-", for examples: Gln-Cys-Gln-Gln-Ala-Val-Gln-Ser-Ala-Val.
Peptide sequences may have additional modifications or molecules at two terminals, with C-carboxyl amidation (-NH2) being a prevalent modification at C-terminus, for examples: Cu-GHK, DIPS-Cu, H-Ile-Pro-Arg-Cys-Arg-Lys-Met-Pro-Gly-Val-Lys-Met-Cys-NH2 and Pam3Cys-Ala-Gly.
And filter RNA or DNA sequences out. RNA or DNA sequences examples are: Hs-GLP-1R-si-3, 5′-UAGAAAUCUAUCUUUGUCCdTdT-3′.

Please list each peptide name and sequence on a separate line, only from the input content, and not prior knowledge. If an explicit peptide name is present but its sequence is absent, report its sequence as "None". If the name of a peptide is absent but its sequence is present, use its sequence as its name. If no peptide sequences are present at all, only shortly output 'None', never output "None: None".
The output format is the peptide name at left and its sequence at right, separated by colon.

Examples of input content and output:


Input content:
Introduction\nThe first synthetic peptide incorporated into skin care products in the late 80s was copper glycine-histidine-lysine (Cu-GHK) generated by Pickard in 1973 [3]. Since then, many short synthetic peptides playing roles in inflammation, extracellular matrix synthesis or pigmentation have been developed. These peptides are used for anti-oxidation, “Botox-like” wrinkle smoothing and collagen stimulation. Cosmeceutical peptides usually have certain features. Copper tripeptides, tetrapeptide PKEK, manganese tripeptide-1, and silk fibroin peptide have been on the market for several decades, but less in vivo efficacy data is available [5]. Their substance mixtures are examined in cosmetic formulations, so as to the actual effect of individual peptides on the skin still remains unclear in many cases. Scientific research on peptides usually focus on identification of functional mechanism, and practical application for cosmetic and/or pharmaceutical use. In the last decade, the development of active peptides has established a new field in cosmeceutical and pharmaceutical skin care. To generate safety profile and functional data of peptide, in vivo animal test was usually chosen in the past. Cell-penetrating peptides (CPP) are peptides that can transport cargos such as chemical compounds, proteins, peptides, and nanoparticles into cells [12]. Most CPP sequences are rich in positively charged residues, and are internalized after interacting with negatively charged glycosaminoglycans (GAGs) and clustering on outer membrane surfaces [13]. For example, a modified PTD (Protein transduction domain) peptide from human immunodeficiency virus (HIV): TAV (RKKRRQRRR) has been shown to have cell membrane penetration property and deliver therapeutic proteins into mammalian cells [14]. Another case is AID (arginine-rich intracellular delivery) peptides which successfully enter and deliver functional proteins into epidermis and dermis of mouse [15,16]. Many modified AID peptides (HGH6, R7) proven to penetrate into skin of living animals with cargos. We examined the inhibitory effect of sevanol on the ASIC1a channel activated by pH 5.5 in the presence of 200 μM FRRFa peptide (Phe-Arg-Arg-Phe-amide), and showed that sevanol and the peptide compete for the binding site.

Formatted output:
* copper glycine-histidine-lysine: Cu-GHK
* tetrapeptide PKEK: PKEK
* manganese tripeptide-1: None
* silk fibroin peptide: None
* TAV: RKKRRQRRR
* HGH6: None
* R7: None
* FRRFa peptide: Phe-Arg-Arg-Phe-amide


Input content:
Based on the provided context information, the following peptide sequences are mentioned:\n\n* Penetratin: CCPP with a sequence not specified\n* NR2B9c: peptide with a sequence of KLSSIESDV. Interestingly, the CARP AIP6 (RLRWR; net charge +3) can inhibit NF-κB activity by an alternative mechanism, by binding to and blocking NF-κβ p65 sub-unit binding to DNA and inhibiting its transcriptional activity (301). The Apolipoprotein E (ApoE) protein derived Ac-hE18A-NH2 (Ac-RKLRKRLLRDWLKAFYDKVAEKLKEAF-NH2; net charge +6) can bind bacterial lipopolysaccharides (LPS) and reduce its inflammatory (e.g., TNF-α, IL-6 production) inducing properties in human blood and primary leukocytes. The COG112 peptide, which comprises AIP6 fused to the penetratin (Table 2) inhibits the inflammatory response in mouse models of pathogen. A Nrf2 amino acid derived sequence (LQLDEETGEFLPIQ) has been developed that disrupts the Nrf2-Keap1 interaction, and when fused to TAT-12 (charge -4.6) has been demonstrated to activate Nrf2.

Formatted output:
* Penetratin: None
* NR2B9c: KLSSIESDV
* AIP6: RLRWR
* Ac-hE18A-NH2: Ac-RKLRKRLLRDWLKAFYDKVAEKLKEAF-NH2
* COG112: None
* LQLDEETGEFLPIQ: LQLDEETGEFLPIQ
* TAT-12: None


Input content:
Overview of CARPs and Neuroprotection Studies\nKey events in the recognition and application of CARPs as neuroprotective agents are summarized in Figure 2. Hexapeptides containing at least two arginine (R) residues at any position as well as one or more lysine (K), tryptophan (W), and cysteine (C) residues displayed ionic current blocking activity. Further analysis revealed that C-carboxyl amidated (-NH2) peptide RRRCWW-NH2 (net charge +3) was capable of blocking NMDA receptor activity. Certain amino acid residues within arginine-rich hexapeptides inhibited the NMDA receptor blocking ability of the peptide (e.g., RFMRNR-NH2; net charge +4, was ineffective), and the neuroprotective action of the peptides was not stereo-selective with L- and D-isoform. Historical time-line for the recognition of CARPs as neuroprotective agents. Other observations concluded that: (i) whereas cationic arginine-rich hexapeptides were highly efficient at blocking NMDA receptor evoked ionic currents, some peptides (e.g., RYYRRW-NH2) blocked AMPA receptor currents by over 60%. In subsequent studies, the peptide RRYCWW-NH2 was demonstrated to antagonize the vanilloid receptor 1 (VR1; also known as the transient receptor potential cation channel subfamily V member 1; TRPV1) mediated currents in a Xenopus expression system and reduce calcium influx in rat dorsal root ganglion neurons following capsaicin or resiniferatoxin VR1 receptor stimulation. CARPs that inhibit Aβ oligomer formation, which is considered neurotoxic include 15M (Ac-VITNPNRRNRTPQMLKR-NH2: net charge 0) (352) and PR-39 peptide (Table 1).

Formatted output:
* RRRCWW-NH2: RRRCWW-NH2
* RW-NH2: RW-NH2
* RFMRNR-NH2: RFMRNR-NH2
* RYYRRW-NH2: RYYRRW-NH2
* RRYCWW-NH2: RRYCWW-NH2
* 15M: Ac-VITNPNRRNRTPQMLKR-NH2
* PR-39: None


Input content:
Experimental\nThe anti-oxidant action of SS peptides has been attributable to the tyrosine and dimethyltyrosine residues, based on the free radial scavenging properties of guanidinium containing molecules it is likely that the arginine residue also contributes to the anti-oxidant property of SS peptides. Anti-oxidant and Free Radical Scavenging Properties: Larger CARP's also have anti-oxidant and lipid peroxidation reducing properties. The lactoferrin derived peptide df8 (GRRRRSVQWCAVSQPEATKCFQWQRNMRKVRGPPVSCIKRDSPIQCIQ; net charge +8.7) has demonstrated anti-oxidant activity in an in vitro free radical scavenging assay. Similarly, the cathelicidin PR-40 (RRRPRPPYLPRPRPPPFFPPRLPPRIPPGFPPRFPPRFP; net charge +12) protects HeLa cells from apoptotic cell death induced by the oxidizing agent tert-butyl hydroperoxide, and inhibits hypoxia induced cell death of endothelial cells. In contrast, arginine-containing penta-peptides peptides (CycK[Myr]RRRRE; Cyc, cyclic peptide; Myr, myristic acid; and myr-KRRRRE; net charge +4) possess methylglyoxal scavenging activity (271), and CycK(Myr)RRRRE prevents methylglyoxal induced pain in mice, and is being considered as a therapy for pain.

Formatted output:
* df8: GRRRRSVQWCAVSQPEATKCFQWQRNMRKVRGPPVSCIKRDSPIQCIQ
* PR-40: RRRPRPPYLPRPRPPPFFPPRLPPRIPPGFPPRFPPRFP
* CycK[Myr]RRRRE: myr-KRRRRE


Input content:
Here a 10-residue peptide, covering major GAG binding motif of a human ribonuclease, is identified as a CPPAIF (anti-inflammatory CPP). CPPAIF has been shown to possess epithelial cell, GAG and lipid binding properties as well as cell penetrating activity through macropinocytosis [20]. Notably, CPPAIF is able to deliver small fluorescent molecules, recombinant proteins, and peptidomimetic drugs into cells [21]. Based on these facts, safety and potential of CPPAIF for cosmeceutical application were examined with skin cell and 3D-skin models following the Organization for Economic Co-operation and Development (OECD) guidelines with special focus on stability, safety, skin irritation, bio-functions and transepidermal activity in this work. CPPAIF, a GAG binding peptide, could penetrate cell membranes with cargos in living animals and was proven to be stable in powder form under room temperature or in water solution under −20 °C.

Formatted output:
None


The current input content:
{content}


Please list the peptide names and their corresponding exact sequences only from the current input content, with the output format above.
"""

no_example_abstract_question=""" 
Peptide sequences are sequences of amino acids and mostly have two types: one uppercase letter (A-Z) for one amino acid and concatenated by hyphen "-". Peptide sequences may have extra modification or molecular letters at the start or end of sequeences. And do not take the name of the peptide as the sequence, such as LL-37, 4F, C16, F4 and so on. And filter RNA or DNA sequences out. RNA or DNA sequences examples are: Hs-GLP-1R-si-3, si-GLP-1R, 5′-UAGAAAUCUAUCUUUGUCCdTdT-3′, 5′-GGACATGTGTTCCAGGAAGGTGTCAGCCATGG-3′.
Please complete the task based solely on the information provided in the given context and do not rely on any prior knowledge or assumptions.
Task: Please make sure to list the names and sequences of any peptides in the following format: 
* peptideName_1: peptideSequence_1
* peptideName_2: peptideSequence_2
If there is no sequence, fill peptideSequence with None and do not fill the peptide name. 

The given current is:
{content}
"""

abstract_question=""" 
Peptide sequences are sequences of amino acids and mostly have two types: one uppercase letter (A-Z) for one amino acid, for examples: YKKHRQRCW, INLKALAALAKKIL; and three letters for one amino acid (Ala, Cys, Asp, Glu, Phe, Gly, His, Ile, Leu, Met, Asn, Pro, Lys, Gln, Arg, Ser, Thr, Val, Trp, Tyr) and concatenated by hyphen "-", for examples: Gln-Cys-Gln-Gln-Ala-Val-Gln-Ser-Ala-Val. Peptide sequences may have extra modification or molecular letters at the start or end of sequeences, for examples: H-Ile-Pro-Arg-Cys-Arg-Lys-Met-Pro-Gly-Val-Lys-Met-Cys-NH2, RRRCYWW-NH2, Pam3Cys-Ala-Gly, etc. And do not take the name of the peptide as the sequence, such as LL-37, 4F, C16, F4 and so on. And filter RNA or DNA sequences out. RNA or DNA sequences examples are: Hs-GLP-1R-si-3, si-GLP-1R, 5′-UAGAAAUCUAUCUUUGUCCdTdT-3′, 5′-GGACATGTGTTCCAGGAAGGTGTCAGCCATGG-3′.
Please complete the task based solely on the information provided in the given context and do not rely on any prior knowledge or assumptions.
Task: Please make sure to list the names and sequences of any peptides in the following format: 
* peptideName_1: peptideSequence_1
* peptideName_2: peptideSequence_2
If there is no sequence, fill peptideSequence with None and do not fill the peptide name. 
For example: 
* B29: None
* Lunasin: SKWQHQQDSCRKQLQGVNLT
* BPCD36: Tyr-Lys-Gly-Lys

The given current is:
{content}
"""

llm_question=""" 
Objective: Analyze the provided content to identify and extract any peptide names and sequences present. Focus on extracting sequences that adhere to the typical format and structure of peptide sequences.

Instructions:
1. Begin by thoroughly examining the given text, paying close attention to any segments that resemble peptide sequence notation, typically typically consisting of uppercase letters from the standard 20-letter amino acid code.
2. Ignore any non-sequence data, such as descriptive text, unless it directly precedes or follows a sequence and aids in confirming its identity as a peptide.
3. The peptide sequences usually have the following two types: one uppercase letter (A-Z) for one amino acid, for examples: YKKHRQRCW, INLKALAALAKKIL; and three letters for one amino acid (Ala, Cys, Asp, Glu, Phe, Gly, His, Ile, Leu, Met, Asn, Pro, Lys, Gln, Arg, Ser, Thr, Val, Trp, Tyr) and concatenated by hyphen "-", for examples: Gln-Cys-Gln-Gln-Ala-Val-Gln-Ser-Ala-Val. Peptide sequences may have extra modification or molecular letters at the start or end of sequeences, for examples: H-Ile-Pro-Arg-Cys-Arg-Lys-Met-Pro-Gly-Val-Lys-Met-Cys-NH2, RRRCYWW-NH2, Pam3Cys-Ala-Gly, etc.
4. Do not take the name of the peptide as the sequence, such as LL-37, 4F, C16, F4 and so on. And filter RNA or DNA sequences out. RNA or DNA sequences examples are: Hs-GLP-1R-si-3, si-GLP-1R, 5′-UAGAAAUCUAUCUUUGUCCdTdT-3′, 5′-GGACATGTGTTCCAGGAAGGTGTCAGCCATGG-3′.
5. Please rely solely on the information provided in the provided context and do not rely on any prior knowledge or assumptions.

Deliverables:
Please make sure to list the names and sequences of any peptides in the following format: 
* peptideName_1: peptideSequence_1
* peptideName_2: peptideSequence_2
If there is no sequence, fill peptideSequence with None and do not fill the peptide name. 
For example: 
* B29: None
* Lunasin: SKWQHQQDSCRKQLQGVNLT
* BPCD36: Tyr-Lys-Gly-Lys

The provided current is:
{content}
"""

QUESTIONS = {
    "shortest_question": shortest_question,
    "concise_question": concise_question,
    "basic_examples_question": basic_examples_question,
    "more_examples_question": more_examples_question,
    "no_examples_question": no_examples_question,
    "full_examples_question": FULL_EXAMPLES_QUESTION,
    "abstract_question": abstract_question,
    "llm_question": llm_question,
    "no_example_abstract_question": no_example_abstract_question
}
# The model auto use sequence as name when name is absent, not need prompts.
# If a peptide name is absent but its sequence is present, use its sequence as its name, for example, "RYYRAW-NH2": "RYYRAW-NH2".


def sanity_check(question=FULL_EXAMPLES_QUESTION):
    example_input_line = "Input content:"
    example_out_line = "Formatted output:"
    lines = question.split("\n")
    input_content = ""
    example_content_count = 0
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith(example_input_line):
            i += 1
            line = lines[i].strip()
            input_content = ""
            while not line.startswith(example_out_line):
                if line:
                    input_content = input_content + line + "\n"
                i += 1
                line = lines[i].strip()
            example_content_count += 1
            # logger.info(f"{content_count} {input_content = }")
            if line.startswith(example_out_line):
                i += 1
                line = lines[i].strip()
                while line.startswith("* "):
                    name, seq = line.split(": ", maxsplit=1)
                    name = name[2:]
                    if name not in input_content:
                        raise Exception(f"{name = } not in {input_content[:100] = }")
                    if seq != "None" and seq not in input_content:
                        raise Exception(
                            f"{name = }, {seq = } not in {input_content[:100] = }"
                        )
                    i += 1
                    line = lines[i].strip()
        else:
            i += 1
    logger.info(f'Examples count {example_content_count}')

if __name__ == "__main__":
    sanity_check()