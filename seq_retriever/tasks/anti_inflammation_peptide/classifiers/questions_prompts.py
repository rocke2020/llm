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


no_example_abstract_question = """ 
Peptide sequences are sequences of amino acids and mostly have two types: one uppercase letter (A-Z) for one amino acid and concatenated by hyphen "-". Peptide sequences may have extra modification or molecular letters at the start or end of sequeences. And do not take the name of the peptide as the sequence, such as LL-37, 4F, C16, F4 and so on. And filter RNA or DNA sequences out. RNA or DNA sequences examples are: Hs-GLP-1R-si-3, si-GLP-1R, 5′-UAGAAAUCUAUCUUUGUCCdTdT-3′, 5′-GGACATGTGTTCCAGGAAGGTGTCAGCCATGG-3′.
Please complete the task based solely on the information provided in the given context and do not rely on any prior knowledge or assumptions.
Task: Please make sure to list the names and sequences of any peptides in the following format: 
* peptideName_1: peptideSequence_1
* peptideName_2: peptideSequence_2
If there is no sequence, fill peptideSequence with None and do not fill the peptide name. 

The given current is:
{content}
"""

abstract_question = """ 
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

llm_question = """ 
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

The provided content is:
{content}
"""


old_anti_question = """
The input content is {content}
Given the input content, please answer the query:
Please determine if peptide sequence {peptide_seq} have the ability of anti-inflammation. 
You should focus on peptide sequence {peptide_seq} and ignore other irrelvant sequence!
Finally, plesae only output "The answer is: 'Yes'" or "The answer is: 'No'", do not output any other content.
"""


old_anti_question_8_4yes = """
Given the input content, please answer the query:
Please determine if peptide sequence have the ability of anti-inflammation. 
You should focus on peptide sequence and ignore other irrelvant sequence!

Examples of input content and output:
The peptide sequence for judging whether it contains anti-inflammatory properties is formyl-Met-Leu-Phe.
Input content:
Indomethacin, a non-steroidal anti-inflammatory drug, competes with the tritiated chemotactic peptide formyl-Met-Leu-Phe (FMLP) for binding to human neutrophils (PMN). This competition which occurs in the concentration range 10(-6)-10(-3) M, with an IC50 of 7 x 10(-5) M, is inversely proportional to the concentration of albumin present in the incubation medium. These data explain why indomethacin is able to inhibit, in the same concentration range and with the same IC50, many of the physiological responses of PMN elicited by the chemotactic peptide formyl-Met-Leu-Phe.
Output:
The answer is: 'Yes'

The peptide sequence for judging whether it contains anti-inflammatory properties is H-Met-Gln-Met-Lys-Lys-Val-Leu-Asp-Ser-OH.
Input content:
The nonapeptide H-Met-Gln-Met-Lys-Lys-Val-Leu-Asp-Ser-OH (antiflammin), corresponding to a high amino acid similarity region of uteroglobin was prepared by solution and solid phase synthesis. The peptide do not inhibit pancreatic phospholipase A2 in vitro and the carrageenin-induced oedema in vivo.
Output:
The answer is: 'Yes'

The peptide sequences for judging whether it contains anti-inflammatory properties are HVFRLKKWIQKVIDQFGE and GKYGFYTHVFRLKKWIQKVIDQFGE.
Input content:
The wound environment is characterized by physiological pH changes. Proteolysis of thrombin by wound-derived proteases, such as neutrophil elastase, generates antimicrobial thrombin-derived C-terminal peptides (TCPs), such as HVF18 (HVFRLKKWIQKVIDQFGE). Presence of such TCPs in human wound fluids in vivo, as well as the occurrence of an evolutionarily conserved His residue in the primary amino acid sequence of TCPs, prompted us to investigate the pH-dependent antibacterial action of HVF18, as well as of the prototypic GKY25 (GKYGFYTHVFRLKKWIQKVIDQFGE). We show that protonation of this His residue at pH 5.5 increases the antibacterial activity of both TCPs against Gram-negative Escherichia coli by membrane disruption. Physiological salt level (150 mM NaCl) augments antibacterial activity of GKY25 but diminishes for the shorter HVF18. Replacing His with Leu or Ser in GKY25 abolishes the His protonation-dependent increase in antibacterial activity at pH 5.5, whereas substitution with Lys maintains activity at neutral (pH 7.4) and acidic pH. Interestingly, both TCPs display decreased binding affinities to human CD14 with decreasing pH, suggesting a likely switch in mode-of-action, from anti-inflammatory at neutral pH to antibacterial at acidic pH. Together, the results demonstrate that apart from structural prerequisites such as peptide length, charge, and hydrophobicity, the evolutionarily conserved His residue of TCPs influences their antibacterial effects and reveals a previously unknown aspect of TCPs biological action.
Output:
The answer is: 'Yes'

The peptide sequences for judging whether it contains anti-inflammatory properties are KLVFF and DAG.
Input content:
Insurmountable blood‒brain barrier (BBB) and complex pathological features are the key factors affecting the treatment of Alzheimer's disease (AD). Poor accumulation of drugs in lesion sites and undesired effectiveness of simply reducing Aβ deposition or TAU protein need to be resolved urgently. Herein, a nanocleaner is designed with a rapamycin-loaded ROS-responsive PLGA core and surface modification with KLVFF peptide and acid-cleavable DAG peptide [R@(ox-PLGA)-KcD]. DAG can enhance the targeting and internalization effect of nanocleaner towards neurovascular unit endothelial cells in AD lesions, and subsequently detach from nanocleaner in response to acidic microenvironment of endosomes to promote the transcytosis of nanocleaner from endothelial cells into brain parenchyma. Then exposed KLVFF can capture and carry Aβ to microglia, attenuating Aβ-induced neurotoxicity. Strikingly, rapamycin, an autophagy promoter, is rapidly liberated from nanocleaner in the high ROS level of lesions to improve Aβ degradation and normalize inflammatory condition. This design altogether accelerates Aβ degradation and alleviates oxidative stress and excessive inflammatory response. Collectively, our finding offers a strategy to target the AD lesions precisely and multi-pronged therapies for clearing the toxic proteins and modulating lesion microenvironment, to achieve efficient AD therapy.
Output:
The answer is: 'Yes'

The peptide sequence for judging whether it contains anti-inflammatory properties is Phe-Leu-Phe-Leu-Phe.
Input content:
Recent investigations conducted with human neutrophils have indicated an involvement for the receptor for formylated peptides, termed FPR, and its analog FPRL1 (or ALXR because it is the receptor for the endogenous ligand lipoxin A(4)) in the in vitro inhibitory actions of the glucocorticoid-regulated protein annexin 1 and its peptidomimetics. To translate these findings in in vivo settings, we have used an ischemia/reperfusion (I/R) procedure to promote leukocyte-endothelium interactions in the mouse mesenteric microcirculation. In naive mice, the annexin 1 mimetic peptide Ac2-26 (20 to 100 microg administered intravenously prior to reperfusion) abolished I/R-induced cell adhesion and emigration, but not cell rolling. In FPR-deficient mice, peptide Ac2-26 retained significant inhibitory actions (about 50% of the effects in naive mice), and these were blocked by an FPR antagonist, termed butyloxycarbonyl-Phe-Leu-Phe-Leu-Phe, or Boc2. In vitro, neutrophils taken from these animals could be activated at high concentrations of formyl-Met-Leu-Phe (30 microM; fMLP), and this effect was blocked by cell incubation with peptide Ac2-26 (66 microM) or Boc2 (100 microM). FPR-deficient neutrophils expressed ALXR mRNA and protein. Both ALXR agonists, lipoxin A(4) and peptide Ac2-26, provoked detachment of adherent leukocytes in naive as well as in FPR-deficient mice, whereas the CXC chemokine KC or fMLP were inactive. The present findings demonstrate that endogenous regulatory autocoids such as lipoxin A(4) and annexin 1-derived peptides function to disengage adherent cells during cell-cell interactions.
Output:
The answer is: 'No'

The peptide sequences for judging whether it contains anti-inflammatory properties are SLIGRL-NH and GYPGQV-NH.
Input content:
Objective: The impact of non-steroidal anti-inflammatory drugs (NSAIDs) to damage the small intestine has been well known. Mica, one kind of natural clay, has been widely marketed in China for the treatment of gastric diseases. However, the role and mechanism of mica in small intestinal injure is still unknown. The study was designed to declare the effects of mica on intestinal injury induced by diclofenac in rats. Methods: Rats were randomly divided into control, model, PAR-2 agonist group (SLIGRL-NH2group), control peptide group (LRGILS-NH2 group), and ERK blocker group (eight mice per group). Morphological changes of mucous membrane of small intestine were observed, and the expression of tryptase, PAR-2, and p-ERK1/2 was measured by immunohistochemistry and western blot. PAR-2 mRNA was tested by qRT-PCR. Rats were also randomly divided into control, model, and mica group (eight mice per group). Morphological changes of mucous membrane were observed. The expression of tryptase, PAR-2, and p-ERK1/2 was measured by immunohistochemistry. Results: The expression of trypsin, PAR-2, and p-ERK1/2 was increased in model group compared with control. The expression of PAR-2 and p-ERK1/2 was increased in SLIGRL-NH2 group compared with model, but not LRGILS-NH2 group. The expression of PAR-2 was down-regulated in ERK blocker group compared with SLIGRL-NH2 group. Macroscopically visible lesions of mucous membrane were positively correlated with the expression of PAR-2 and p-ERK1/2. Furthermore, we also found that mica could inhibit small intestinal injure, as evidenced by the improvement of macroscopic damage. Tryptase, PAR-2, and p-ERK1/2 expression was down-regulated in mica group compared with model group. Conclusion: Mica inhibit small intestinal injury induced by NSAIDs via ERK signaling pathway.
Output:
The answer is: 'No'

The peptide sequence for judging whether it contains anti-inflammatory properties is SFLLRN.
Input content:
Vorapaxar had no effect on PFCC, ADP- or collagen-induced PA, thrombin time, fibrinogen, PT, PTT, von Willebrand factor (vWF), D-dimer, or endothelial function (P > .05 in all groups). Inhibition of SFLLRN (PAR-1 activating peptide)-stimulated PA by vorapaxar was accelerated by A + C at 2 hours (P < .05 versus other groups) with nearly complete inhibition by 30 days that persisted through 30 days after discontinuation in all groups (P < .001). SFLLRN-induced PA during offset was lower in APT patients versus APT-naïve patients (P < .05). Inhibition of UTxB2 was observed in APT-naive patients treated with vorapaxar (P < .05). Minor bleeding was only observed in C-treated patients.
Output:
The answer is: 'No'

The peptide sequences for judging whether it contains anti-inflammatory properties are SAKELR, MLRQTR, HASILP, KKEPWI and CLDPKENWVQRVVEKFLKRAENS.
Input content:
The human CXCL8 plays important roles in inflammation by activation of neutrophils through the hCXCR1 and hCXCR2 receptors. The role of hCXCR1 and hCXCR2 in the pathogenesis of inflammatory responses has encouraged the development of antagonists of these receptors. In this study, we used phage display peptide libraries to identify peptides antagonists that block the interactions between hCXCL8 and hCXCR1/2. Two linear hexapeptides (MSRAKE and CAKELR) and two disulfide-constrained hexapeptides (CLRSGRFC and CLPWKENC) were recovered by panning phage libraries on hCXCR1- and hCXCR2-transfected murine pre-B cells after specific elution with hCXCL8. Sequence analysis revealed homology between the linear hexapeptides and the N-terminal domain (1-SAKELR-6), whereas the constrained peptides are composed of non-contiguous amino acids mimicking spatial structure on the surface of folded C-terminal portion of hCXCL8 (50-CLDPKENWVQRVVEKFLKRAENS-72). The synthetic linear and structurally constrained peptides competed for (125)I-hCXCL8 binding to hCXCR1 and hCXCR2 (IC(50) comprised between 10 and 100μM). Furthermore, they inhibited the intracellular calcium flux and the migration of hCXCR1/hCXCR2 transfectants; and desensitized hCXCR1 and hCXCR2 receptors on neutrophils, reducing their chemotactic responses induced by ELR-CXC chemokines (hCXCL8, hCXCL1, hCXCL2, hCXCL3, and hCXCL5). To better characterize the residues required for hCXCL8 binding, we identified three linear peptides MLRQTR, HASILP and KKEPWI specific to hCXCL8. These peptides similarly displaced the binding of (125)I-hCXCL8 to hCXCR1 (IC(50) ranging from 8.5 to 10μM) in a dose-dependent manner, inhibited hCXCL8 induced increases in the intracellular calcium, and migration of hCXCR1- and hCXCR2-transfected cells. The identified peptides could be used as antagonists of hCXCL8-induced activities related to its interaction with hCXCR1 and hCXCR2 receptors and may help in the design of new anti-inflammatory therapeutic molecules.
Output:
The answer is: 'No'


Now, learn these sample information and determine whether the peptide sequence {peptide_seq} contains anti-inflammatory properties.
You should focus on peptide sequence {peptide_seq} and ignore other irrelvant sequence!
The current input content: {content}
Finally, plesae only output "The answer is: 'Yes'" or "The answer is: 'No'", do not output any other content.
"""


new_anti_question = """
The input content is: {content}
Given the input context, please answer the query:
Please score the anti-inflammatory properties of peptide sequence {peptide_seq} on a scale of 0-100, where 0 represents that the peptide cannot have anti-inflammatory properties, and 100 represents that the peptide definitely contains anti-inflammatory properties.
Also, provide corresponding analysis reasons.
Please make sure to list the anti-inflammatory score and corresponding reason in the following format:
* Anti-inflammatory score: 80
* Corresponding reason: reasons
"""


score_anti_question_8_4yes = """
Given the input context, please answer the query:
Please score the anti-inflammatory properties of peptide sequence on a scale of 0-100, where 0 represents that the peptide cannot have anti-inflammatory properties, and 100 represents that the peptide definitely contains anti-inflammatory properties.
Also, provide corresponding analysis reasons.
Please make sure to list the anti-inflammatory score and corresponding reason.

Examples of input content and output:
The peptide sequence for judging whether it contains anti-inflammatory properties is formyl-Met-Leu-Phe.
Input content:
Indomethacin, a non-steroidal anti-inflammatory drug, competes with the tritiated chemotactic peptide formyl-Met-Leu-Phe (FMLP) for binding to human neutrophils (PMN). This competition which occurs in the concentration range 10(-6)-10(-3) M, with an IC50 of 7 x 10(-5) M, is inversely proportional to the concentration of albumin present in the incubation medium. These data explain why indomethacin is able to inhibit, in the same concentration range and with the same IC50, many of the physiological responses of PMN elicited by the chemotactic peptide formyl-Met-Leu-Phe.
Formatted output:
* Anti-inflammatory score: 100
* Corresponding reason: please provide the relevant reasons.

The peptide sequence for judging whether it contains anti-inflammatory properties is H-Met-Gln-Met-Lys-Lys-Val-Leu-Asp-Ser-OH.
Input content:
The nonapeptide H-Met-Gln-Met-Lys-Lys-Val-Leu-Asp-Ser-OH (antiflammin), corresponding to a high amino acid similarity region of uteroglobin was prepared by solution and solid phase synthesis. The peptide do not inhibit pancreatic phospholipase A2 in vitro and the carrageenin-induced oedema in vivo.
Formatted output:
* Anti-inflammatory score: 100
* Corresponding reason: please provide the relevant reasons.

The peptide sequences for judging whether it contains anti-inflammatory properties are HVFRLKKWIQKVIDQFGE and GKYGFYTHVFRLKKWIQKVIDQFGE.
Input content:
The wound environment is characterized by physiological pH changes. Proteolysis of thrombin by wound-derived proteases, such as neutrophil elastase, generates antimicrobial thrombin-derived C-terminal peptides (TCPs), such as HVF18 (HVFRLKKWIQKVIDQFGE). Presence of such TCPs in human wound fluids in vivo, as well as the occurrence of an evolutionarily conserved His residue in the primary amino acid sequence of TCPs, prompted us to investigate the pH-dependent antibacterial action of HVF18, as well as of the prototypic GKY25 (GKYGFYTHVFRLKKWIQKVIDQFGE). We show that protonation of this His residue at pH 5.5 increases the antibacterial activity of both TCPs against Gram-negative Escherichia coli by membrane disruption. Physiological salt level (150 mM NaCl) augments antibacterial activity of GKY25 but diminishes for the shorter HVF18. Replacing His with Leu or Ser in GKY25 abolishes the His protonation-dependent increase in antibacterial activity at pH 5.5, whereas substitution with Lys maintains activity at neutral (pH 7.4) and acidic pH. Interestingly, both TCPs display decreased binding affinities to human CD14 with decreasing pH, suggesting a likely switch in mode-of-action, from anti-inflammatory at neutral pH to antibacterial at acidic pH. Together, the results demonstrate that apart from structural prerequisites such as peptide length, charge, and hydrophobicity, the evolutionarily conserved His residue of TCPs influences their antibacterial effects and reveals a previously unknown aspect of TCPs biological action.
Formatted output:
* Anti-inflammatory score: 100
* Corresponding reason: please provide the relevant reasons.

The peptide sequences for judging whether it contains anti-inflammatory properties are KLVFF and DAG.
Input content:
Insurmountable blood‒brain barrier (BBB) and complex pathological features are the key factors affecting the treatment of Alzheimer's disease (AD). Poor accumulation of drugs in lesion sites and undesired effectiveness of simply reducing Aβ deposition or TAU protein need to be resolved urgently. Herein, a nanocleaner is designed with a rapamycin-loaded ROS-responsive PLGA core and surface modification with KLVFF peptide and acid-cleavable DAG peptide [R@(ox-PLGA)-KcD]. DAG can enhance the targeting and internalization effect of nanocleaner towards neurovascular unit endothelial cells in AD lesions, and subsequently detach from nanocleaner in response to acidic microenvironment of endosomes to promote the transcytosis of nanocleaner from endothelial cells into brain parenchyma. Then exposed KLVFF can capture and carry Aβ to microglia, attenuating Aβ-induced neurotoxicity. Strikingly, rapamycin, an autophagy promoter, is rapidly liberated from nanocleaner in the high ROS level of lesions to improve Aβ degradation and normalize inflammatory condition. This design altogether accelerates Aβ degradation and alleviates oxidative stress and excessive inflammatory response. Collectively, our finding offers a strategy to target the AD lesions precisely and multi-pronged therapies for clearing the toxic proteins and modulating lesion microenvironment, to achieve efficient AD therapy.
Formatted output:
* Anti-inflammatory score: 100
* Corresponding reason: please provide the relevant reasons.

The peptide sequence for judging whether it contains anti-inflammatory properties is Phe-Leu-Phe-Leu-Phe.
Input content:
Recent investigations conducted with human neutrophils have indicated an involvement for the receptor for formylated peptides, termed FPR, and its analog FPRL1 (or ALXR because it is the receptor for the endogenous ligand lipoxin A(4)) in the in vitro inhibitory actions of the glucocorticoid-regulated protein annexin 1 and its peptidomimetics. To translate these findings in in vivo settings, we have used an ischemia/reperfusion (I/R) procedure to promote leukocyte-endothelium interactions in the mouse mesenteric microcirculation. In naive mice, the annexin 1 mimetic peptide Ac2-26 (20 to 100 microg administered intravenously prior to reperfusion) abolished I/R-induced cell adhesion and emigration, but not cell rolling. In FPR-deficient mice, peptide Ac2-26 retained significant inhibitory actions (about 50% of the effects in naive mice), and these were blocked by an FPR antagonist, termed butyloxycarbonyl-Phe-Leu-Phe-Leu-Phe, or Boc2. In vitro, neutrophils taken from these animals could be activated at high concentrations of formyl-Met-Leu-Phe (30 microM; fMLP), and this effect was blocked by cell incubation with peptide Ac2-26 (66 microM) or Boc2 (100 microM). FPR-deficient neutrophils expressed ALXR mRNA and protein. Both ALXR agonists, lipoxin A(4) and peptide Ac2-26, provoked detachment of adherent leukocytes in naive as well as in FPR-deficient mice, whereas the CXC chemokine KC or fMLP were inactive. The present findings demonstrate that endogenous regulatory autocoids such as lipoxin A(4) and annexin 1-derived peptides function to disengage adherent cells during cell-cell interactions.
Formatted output:
* Anti-inflammatory score: 0
* Corresponding reason: please provide the relevant reasons.

The peptide sequences for judging whether it contains anti-inflammatory properties are SLIGRL-NH and GYPGQV-NH.
Input content:
Objective: The impact of non-steroidal anti-inflammatory drugs (NSAIDs) to damage the small intestine has been well known. Mica, one kind of natural clay, has been widely marketed in China for the treatment of gastric diseases. However, the role and mechanism of mica in small intestinal injure is still unknown. The study was designed to declare the effects of mica on intestinal injury induced by diclofenac in rats. Methods: Rats were randomly divided into control, model, PAR-2 agonist group (SLIGRL-NH2group), control peptide group (LRGILS-NH2 group), and ERK blocker group (eight mice per group). Morphological changes of mucous membrane of small intestine were observed, and the expression of tryptase, PAR-2, and p-ERK1/2 was measured by immunohistochemistry and western blot. PAR-2 mRNA was tested by qRT-PCR. Rats were also randomly divided into control, model, and mica group (eight mice per group). Morphological changes of mucous membrane were observed. The expression of tryptase, PAR-2, and p-ERK1/2 was measured by immunohistochemistry. Results: The expression of trypsin, PAR-2, and p-ERK1/2 was increased in model group compared with control. The expression of PAR-2 and p-ERK1/2 was increased in SLIGRL-NH2 group compared with model, but not LRGILS-NH2 group. The expression of PAR-2 was down-regulated in ERK blocker group compared with SLIGRL-NH2 group. Macroscopically visible lesions of mucous membrane were positively correlated with the expression of PAR-2 and p-ERK1/2. Furthermore, we also found that mica could inhibit small intestinal injure, as evidenced by the improvement of macroscopic damage. Tryptase, PAR-2, and p-ERK1/2 expression was down-regulated in mica group compared with model group. Conclusion: Mica inhibit small intestinal injury induced by NSAIDs via ERK signaling pathway.
Formatted output:
* Anti-inflammatory score: 0
* Corresponding reason: please provide the relevant reasons.

The peptide sequence for judging whether it contains anti-inflammatory properties is SFLLRN.
Input content:
Vorapaxar had no effect on PFCC, ADP- or collagen-induced PA, thrombin time, fibrinogen, PT, PTT, von Willebrand factor (vWF), D-dimer, or endothelial function (P > .05 in all groups). Inhibition of SFLLRN (PAR-1 activating peptide)-stimulated PA by vorapaxar was accelerated by A + C at 2 hours (P < .05 versus other groups) with nearly complete inhibition by 30 days that persisted through 30 days after discontinuation in all groups (P < .001). SFLLRN-induced PA during offset was lower in APT patients versus APT-naïve patients (P < .05). Inhibition of UTxB2 was observed in APT-naive patients treated with vorapaxar (P < .05). Minor bleeding was only observed in C-treated patients.
Formatted output:
* Anti-inflammatory score: 0
* Corresponding reason: please provide the relevant reasons.

The peptide sequences for judging whether it contains anti-inflammatory properties are SAKELR, MLRQTR, HASILP, KKEPWI and CLDPKENWVQRVVEKFLKRAENS.
Input content:
The human CXCL8 plays important roles in inflammation by activation of neutrophils through the hCXCR1 and hCXCR2 receptors. The role of hCXCR1 and hCXCR2 in the pathogenesis of inflammatory responses has encouraged the development of antagonists of these receptors. In this study, we used phage display peptide libraries to identify peptides antagonists that block the interactions between hCXCL8 and hCXCR1/2. Two linear hexapeptides (MSRAKE and CAKELR) and two disulfide-constrained hexapeptides (CLRSGRFC and CLPWKENC) were recovered by panning phage libraries on hCXCR1- and hCXCR2-transfected murine pre-B cells after specific elution with hCXCL8. Sequence analysis revealed homology between the linear hexapeptides and the N-terminal domain (1-SAKELR-6), whereas the constrained peptides are composed of non-contiguous amino acids mimicking spatial structure on the surface of folded C-terminal portion of hCXCL8 (50-CLDPKENWVQRVVEKFLKRAENS-72). The synthetic linear and structurally constrained peptides competed for (125)I-hCXCL8 binding to hCXCR1 and hCXCR2 (IC(50) comprised between 10 and 100μM). Furthermore, they inhibited the intracellular calcium flux and the migration of hCXCR1/hCXCR2 transfectants; and desensitized hCXCR1 and hCXCR2 receptors on neutrophils, reducing their chemotactic responses induced by ELR-CXC chemokines (hCXCL8, hCXCL1, hCXCL2, hCXCL3, and hCXCL5). To better characterize the residues required for hCXCL8 binding, we identified three linear peptides MLRQTR, HASILP and KKEPWI specific to hCXCL8. These peptides similarly displaced the binding of (125)I-hCXCL8 to hCXCR1 (IC(50) ranging from 8.5 to 10μM) in a dose-dependent manner, inhibited hCXCL8 induced increases in the intracellular calcium, and migration of hCXCR1- and hCXCR2-transfected cells. The identified peptides could be used as antagonists of hCXCL8-induced activities related to its interaction with hCXCR1 and hCXCR2 receptors and may help in the design of new anti-inflammatory therapeutic molecules.
Formatted output:
The answer is: 'No'
* Anti-inflammatory score: 0
* Corresponding reason: please provide the relevant reasons.



Now, learn these correct sample information and determine whether the peptide sequence {peptide_seq} contains anti-inflammatory properties.
You should focus on peptide sequence {peptide_seq} and ignore other irrelvant sequence!
Please do not answer with N/A.
The current input content: {content}
Finally, plesae only print formatted output, do not output any other content.
"""


QUESTIONS = {
    "shortest_question": shortest_question,
    "concise_question": concise_question,
    "basic_examples_question": basic_examples_question,
    "more_examples_question": more_examples_question,
    "no_examples_question": no_examples_question,
    "abstract_question": abstract_question,
    "llm_question": llm_question,
    "no_example_abstract_question": no_example_abstract_question,
    "old_anti_question": old_anti_question,
    "old_anti_question_8_4yes": old_anti_question_8_4yes,
    "new_anti_question": new_anti_question,
    "new_anti_question_8_4yes": score_anti_question_8_4yes,
}
# The model auto use sequence as name when name is absent, not need prompts.
# If a peptide name is absent but its sequence is present, use its sequence as its name, for example, "RYYRAW-NH2": "RYYRAW-NH2".


def sanity_check(question):
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
    logger.info(f"Examples count {example_content_count}")
