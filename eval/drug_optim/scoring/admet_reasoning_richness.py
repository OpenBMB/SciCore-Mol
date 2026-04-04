# -*- coding: utf-8 -*-
# from sklearn.metrics import f1_score
try:
 from sklearn.metrics import f1_score
except ImportError:
 def f1_score(y_true, y_pred):
 """F1implement"""
 if len(y_true) != len(y_pred):
 return 0.0
 
 tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
 fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
 fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
 
 if tp + fp == 0: # 
 precision = 0.0
 else:
 precision = tp / (tp + fp)
 
 if tp + fn == 0: # 
 recall = 0.0
 else:
 recall = tp / (tp + fn)
 
 if precision + recall == 0:
 return 0.0
 else:
 return 2 * (precision * recall) / (precision + recall)
import json
import time
import re
import os
from typing import Dict, Any, List, Tuple, Set, Optional
from .test2 import search_admet
from .filter import checks, check_item

# LMS Judgeremove - use lms_judge

def simple_reasoning_quality_evaluation(reasoning: str) -> Dict[str, float]:
 """
 inferenceevaluationdependencyexternalAPI
 based onfeaturekeyanalysis
 """
 if not reasoning or not reasoning.strip():
 return {
 "problem_identification": 0.0,
 "solution_quality": 0.0,
 "chemical_knowledge": 0.0,
 "optimization_strategy": 0.0,
 "overall_quality": 0.0,
 "lms_judge_available": False
 }
 
 reasoning_lower = reasoning.lower()
 
 # recognitionkey
 problem_keywords = [
 "", "issue", "problem", "", "deficiency", "", "lack",
 "", "toxicity", "", "toxic", "", "side effect",
 "", "metabolism", "", "metabolic stability",
 "", "permeability", "", "solubility",
 "clear", "clearance", "", "half-life"
 ]
 
 # key
 solution_keywords = [
 "optimization", "optimize", "", "improve", "", "enhance",
 "", "reduce", "", "decrease", "", "increase",
 "", "modify", "replace", "replace", "", "add",
 "", "remove", "", "avoid", "", "prevent"
 ]
 
 # key
 chemical_keywords = [
 "", "functional group", "", "group", "", "atom",
 "key", "bond", "", "ring", "", "aromatic", "", "aliphatic",
 "", "polar", "", "nonpolar", "key", "hydrogen bond",
 "molecule", "molecular weight", "logp", "logd", "tpsa"
 ]
 
 # optimizationstrategykey
 strategy_keywords = [
 "strategy", "strategy", "method", "method", "", "approach",
 "", "design", "", "synthesis", "", "modification",
 "", "structure", "", "structure-activity",
 "", "pharmacophore", "", "lead compound"
 ]
 
 def calculate_score(text, keywords):
 """computekeymatch"""
 matches = sum(1 for keyword in keywords if keyword in text)
 return min(matches / len(keywords) * 2, 1.0) # 0-1
 
 # compute
 problem_score = calculate_score(reasoning_lower, problem_keywords)
 solution_score = calculate_score(reasoning_lower, solution_keywords)
 chemical_score = calculate_score(reasoning_lower, chemical_keywords)
 strategy_score = calculate_score(reasoning_lower, strategy_keywords)
 
 # length
 length_bonus = min(len(reasoning) / 500, 0.2) # 0.2
 
 # computetotal
 overall_score = (problem_score + solution_score + chemical_score + strategy_score) / 4 + length_bonus
 overall_score = min(overall_score, 1.0)
 
 return {
 "problem_identification": problem_score,
 "solution_quality": solution_score,
 "chemical_knowledge": chemical_score,
 "optimization_strategy": strategy_score,
 "overall_quality": overall_score,
 "lms_judge_available": False,
 "evaluation_method": "simple_keyword_analysis"
 }

############################################################
# 1) config
############################################################

# —— 119 ADMET metricoriginalmerge——
FEATURES = [
 "10", "Alarm_NMR Rule", "Acute Toxicity Rule", "AMES Toxicity", "Aquatic Toxicity Rule",
 "A549 Cytotoxicity", "BCF", "BCRP inhibitor", "BMS Rule", "BBB", "Boiling point",
 "Caco-2 Permeability", "Carcinogenicity", "Chelating Rule", "CL_plasma",
 "CYP1A2 inhibitor", "CYP1A2 substrate", "CYP2B6 inhibitor", "CYP2B6 substrate",
 "CYP2C19 inhibitor", "CYP2C19 substrate", "CYP2C8 inhibitor", "CYP2C9 inhibitor",
 "CYP2C9 substrate", "CYP2D6 inhibitor", "CYP2D6 substrate", "CYP3A4 inhibitor",
 "CYP3A4 substrate", "DILI", "Density", "Eye Corrosion", "Eye Irritation",
 "FDAMDD", "FLuc inhibitors", "Flexibility", "Fsp3", "F20%", "F30%", "F50%",
 "Fu", "GASA", "GSK Rule", "GoldenTriangle", "Genotoxic Carcinogenicity Mutagenicity Rule",
 "Genotoxicity", "Green fluorescence", "HBD (nHD)", "HBA (nHA)",
 "Hek293 Cytotoxicity", "HIA", "HLM Stability", "Human Hepatotoxicity",
 "IG C50", "IGC50", "LC50DM", "LC50FM", "Lipinski Rule", "logD7.4", "logP",
 "logS", "MCE-18", "MDCK Permeability", "Melting point", "Molecular Weight (MW)",
 "NPscore", "NonBiodegradable", "NonGenotoxic Carcinogenicity Rule", "nHet",
 "nHD", "nHA", "nRing", "nRig", "nRot", "NR-AR", "NR-AR-LBD", "NR-AhR",
 "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "PAINS", "PAMPA",
 "Pfizer Rule", "PPB", "Promiscuous compounds", "QED", "Rat Oral Acute Toxicity",
 "Respiratory", "RPMI-8226 Immunitoxicity", "Reactive compounds", "Roam (nRot)",
 "SAscore", "Skin Sensitization", "Skin Sensitization Rule", "Stereo Centers",
 "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53", "Topological Polar Surface Area (TPSA)",
 "T1/2", "Toxicity: Eye Corrosion", "Toxicity: Eye Irritation",
 "Toxicity: Human Hepatotoxicity", "Toxicity: Drug-induced Nephrotoxicity",
 "Toxicity: Drug-induced Neurotoxicity", "Toxicity: Ototoxicity",
 "Toxicity: Hematotoxicity", "Toxicity: Genotoxicity", "Toxicity: Carcinogenicity",
 "Toxicity: Skin Sensitization", "Toxicity: DILI", "Toxicity: AMES Toxicity",
 "Toxicity: Rats Oral Acute Toxicity", "Toxicity: FDAMDD",
 "Toxicity: RPMI-8226 Immunitoxicity", "Toxicity: A549 Cytotoxicity",
 "Toxicity: Hek293 Cytotoxicity", "Toxicity: BCF", "Toxicity: IGC50",
 "Toxicity: LC50DM", "Toxicity: LC50FM"
]

# —— mapping (ALIAS) ——
ALIAS: Dict[str, Set[str]] = {
 "10": {"10", "ten peptides"},
 "Alarm_NMR Rule": {"alarm_nmr rule", " nmr rule"},
 "Acute Toxicity Rule": {"acute toxicity rule", " rule"},
 "AMES Toxicity": {"ames toxicity", "ames", " ", "mutagenicity"},
 "Aquatic Toxicity Rule": {"aquatic toxicity rule", " rule"},
 "A549 Cytotoxicity": {"a549 cytotoxicity", "a549 "},
 "BCF": {"bcf", ""},
 "BCRP inhibitor": {"bcrp inhibitor", "bcrp ", "bcrp"},
 "BMS Rule": {"bms rule", "bms rule"},
 "BBB": {"bbb", "blood brain barrier", ""},
 "Boiling point": {"boiling point", ""},
 "Caco-2 Permeability": {"caco-2 permeability", "caco2", "caco-2", "caco2 permeability"},
 "Carcinogenicity": {"carcinogenicity", ""},
 "Chelating Rule": {"chelating rule", " rule"},
 "CL_plasma": {"cl_plasma", "cl plasma", "cl<sub>plasma</sub>", " clear"},
 "CYP1A2 inhibitor": {"cyp1a2 inhibitor", "cyp1a2 ", " cyp1a2"},
 "CYP1A2 substrate": {"cyp1a2 substrate", "cyp1a2 "},
 "CYP2B6 inhibitor": {"cyp2b6 inhibitor", "cyp2b6 "},
 "CYP2B6 substrate": {"cyp2b6 substrate", "cyp2b6 "},
 "CYP2C19 inhibitor": {"cyp2c19 inhibitor", "cyp2c19 "},
 "CYP2C19 substrate": {"cyp2c19 substrate", "cyp2c19 "},
 "CYP2C8 inhibitor": {"cyp2c8 inhibitor", "cyp2c8 "},
 "CYP2C9 inhibitor": {"cyp2c9 inhibitor", "cyp2c9 "},
 "CYP2C9 substrate": {"cyp2c9 substrate", "cyp2c9 "},
 "CYP2D6 inhibitor": {"cyp2d6 inhibitor", "cyp2d6 ", " cyp2d6", "cyp2d6/"},
 "CYP2D6 substrate": {"cyp2d6 substrate", "cyp2d6 "},
 "CYP3A4 inhibitor": {"cyp3a4 inhibitor", "cyp3a4 ", " cyp3a4", "cyp3a4/"},
 "CYP3A4 substrate": {"cyp3a4 substrate", "cyp3a4 "},
 "DILI": {"dili", "drug-induced liver injury", "drug ", "drug "},
 "Density": {"density", ""},
 "Eye Corrosion": {"eye corrosion", ""},
 "Eye Irritation": {"eye irritation", ""},
 "FDAMDD": {"fdamdd", "fda max", "fda mdd"},
 "FLuc inhibitors": {"fluc inhibitors", " "},
 "Flexibility": {"flexibility", ""},
 "Fsp3": {"fsp3", "fsp³", "fsp 3", "fsp<sup>3</sup>"},
 "F20%": {"f20%"},
 "F30%": {"f30%"},
 "F50%": {"f50%"},
 "Fu": {"fu", " "},
 "GASA": {"gasa"},
 "GSK Rule": {"gsk rule", "gsk rule"},
 "GoldenTriangle": {"goldentriangle", "rule"},
 "Genotoxic Carcinogenicity Mutagenicity Rule": {
 "genotoxic carcinogenicity mutagenicity rule", " rule"
 },
 "Genotoxicity": {"genotoxicity", "", "genotox"},
 "Green fluorescence": {"green fluorescence", ""},
 "HBD (nHD)": {"hbd (nhd)", "hbd", "nhd", ""},
 "HBA (nHA)": {"hba (nha)", "hba", "nha", ""},
 "Hek293 Cytotoxicity": {"hek293 cytotoxicity", "hek293 "},
 "HIA": {"hia", "human intestinal absorption", ""},
 "HLM Stability": {"hlm stability", " ", " ", " ", "molecule "},
 "Human Hepatotoxicity": {"human hepatotoxicity", "class ", "liver toxicity"},
 "IG C50": {"ig c50", "igc50"},
 "IGC50": {"igc50"},
 "LC50DM": {"lc50dm"},
 "LC50FM": {"lc50fm"},
 "Lipinski Rule": {"lipinski rule", "lipinski rule"},
 "logD7.4": {"logd7.4", "logd 7.4", "logd_7_4"},
 "logP": {"logp", "clogp"},
 "logS": {"logs", ""},
 "MCE-18": {"mce-18"},
 "MDCK Permeability": {"mdck permeability", "mdck"},
 "Melting point": {"melting point", ""},
 "Molecular Weight (MW)": {"molecular weight (mw)", "molecular weight", "mw", "molecule"},
 "NPscore": {"npscore", "np "},
 "NonBiodegradable": {"nonbiodegradable", " "},
 "NonGenotoxic Carcinogenicity Rule": {"nongenotoxic carcinogenicity rule", " rule"},
 "nHet": {"nhet", ""},
 "nHD": {"nhd", " "},
 "nHA": {"nha", " "},
 "nRing": {"nring", ""},
 "nRig": {"nrig", " "},
 "nRot": {"nrot", "key ", "roam"},
 "NR-AR": {"nr-ar"},
 "NR-AR-LBD": {"nr-ar-lbd"},
 "NR-AhR": {"nr-ahr"},
 "NR-Aromatase": {"nr-aromatase", "nr-"},
 "NR-ER": {"nr-er"},
 "NR-ER-LBD": {"nr-er-lbd"},
 "NR-PPAR-gamma": {"nr-ppar-gamma", "nr-ppar-γ"},
 "PAINS": {"pains"},
 "PAMPA": {"pampa"},
 "Pfizer Rule": {"pfizer rule", "pfizer rule"},
 "PPB": {"ppb", "plasma protein binding", " "},
 "Promiscuous compounds": {"promiscuous compounds", " "},
 "QED": {"qed", "drug similar"},
 "Rat Oral Acute Toxicity": {"rat oral acute toxicity", " "},
 "Respiratory": {"respiratory", ""},
 "RPMI-8226 Immunitoxicity": {"rpmi-8226 immunitoxicity", "rpmi-8226 "},
 "Reactive compounds": {"reactive compounds", ""},
 "SAscore": {"sascore", " ", "synthesizability"},
 "Skin Sensitization": {"skin sensitization", " "},
 "Skin Sensitization Rule": {"skin sensitization rule", " rule"},
 "Stereo Centers": {"stereo centers", " "},
 "SR-ARE": {"sr-are"},
 "SR-ATAD5": {"sr-atad5"},
 "SR-HSE": {"sr-hse"},
 "SR-MMP": {"sr-mmp"},
 "SR-p53": {"sr-p53"},
 "Topological Polar Surface Area (TPSA)": {"topological polar surface area (tpsa)", "tpsa", ""},
 "T1/2": {"t1/2", ""},
 # Toxicity 
 "Toxicity: Eye Corrosion": {"toxicity: eye corrosion", ""},
 "Toxicity: Eye Irritation": {"toxicity: eye irritation", ""},
 "Toxicity: Human Hepatotoxicity": {"toxicity: human hepatotoxicity", "class "},
 "Toxicity: Drug-induced Nephrotoxicity": {"toxicity: drug-induced nephrotoxicity", "drug "},
 "Toxicity: Drug-induced Neurotoxicity": {"toxicity: drug-induced neurotoxicity", "drug "},
 "Toxicity: Ototoxicity": {"toxicity: ototoxicity", ""},
 "Toxicity: Hematotoxicity": {"toxicity: hematotoxicity", " "},
 "Toxicity: Genotoxicity": {"toxicity: genotoxicity", " "},
 "Toxicity: Carcinogenicity": {"toxicity: carcinogenicity", ""},
 "Toxicity: Skin Sensitization": {"toxicity: skin sensitization", " "},
 "Toxicity: DILI": {"toxicity: dili", "drug "},
 "Toxicity: AMES Toxicity": {"toxicity: ames toxicity", "ames "},
 "Toxicity: Rats Oral Acute Toxicity": {"toxicity: rats oral acute toxicity", " "},
 "Toxicity: FDAMDD": {"toxicity: fdamdd", "fda mdd"},
 "Toxicity: RPMI-8226 Immunitoxicity": {"toxicity: rpmi-8226 immunitoxicity", "rpmi-8226 "},
 "Toxicity: A549 Cytotoxicity": {"toxicity: a549 cytotoxicity", "a549 "},
 "Toxicity: Hek293 Cytotoxicity": {"toxicity: hek293 cytotoxicity", "hek293 "},
 "Toxicity: BCF": {"toxicity: bcf"},
 "Toxicity: IGC50": {"toxicity: igc50"},
 "Toxicity: LC50DM": {"toxicity: lc50dm"},
 "Toxicity: LC50FM": {"toxicity: lc50fm"},
}
# CYP ""
ALIAS.setdefault("CYP3A4 inhibitor", set()).update({"cyp3a inhibitor", "cyp3a4 inhibitor", "cyp3a "})
ALIAS.setdefault("CYP1A2 inhibitor", set()).update({"cyp1a inhibitor", "cyp1a2 inhibitor", "cyp1a "})

# logP 
ALIAS.setdefault("logP", set()).update({"clogp", "xlogp", "xlogp3", "log p", "ilogp"})

# Fu
ALIAS.setdefault("Fu", set()).update({
 "fraction unbound", "unbound fraction", "fu_plasma", "fu (plasma)", "fu%", "fu - plasma"
})

# Flexibility Rotatable Bonds / nRot
ALIAS.setdefault("Flexibility", set()).update({
 "rotatable bonds", "num rotatable bonds", "nrot", "roam", "key "
})
# ALIAS.setdefault
ALIAS.update({
 #
 "DILI": {"dili", "drug-induced liver injury", "drug ", "drug ", 
 "hepatotoxicity", "liver injury", "liver toxicity", "hepatic toxicity", 
 "liver damage", "hepatotoxic", "liver safety", "hepatic injury"},
 
 "Human Hepatotoxicity": {"human hepatotoxicity", "class ", "liver toxicity",
 "hepatotoxicity", "hepatic toxicity", "liver injury", 
 "liver damage", "hepatotoxic", "liver safety"},
 
 #
 "Genotoxicity": {"genotoxicity", "", "genotox", "genotoxic", "mutagenicity",
 "mutagenic", "genetic toxicity", "DNA damage", "chromosomal damage"},
 
 "AMES Toxicity": {"ames toxicity", "ames", " ", "mutagenicity", "mutagenic",
 "ames test", "bacterial mutagenicity", "in vitro mutagenicity"},
 
 #
 "HLM Stability": {"hlm stability", " ", " ", " ", 
 "molecule ", "metabolic stability", "metabolism stability",
 "hepatic metabolism", "liver metabolism", "metabolic clearance",
 "oxidative metabolism", "metabolic activation"},
 
 #
 "Caco-2 Permeability": {"caco-2 permeability", "caco2", "caco-2", "caco2 permeability",
 "intestinal permeability", "permeability", "membrane permeability",
 "passive permeability", "absorption", "bioavailability"},
 
 #
 "Toxicity: Drug-induced Neurotoxicity": {"toxicity: drug-induced neurotoxicity", 
 "drug ", "neurotoxicity", "neurotoxic",
 "CNS toxicity", "brain toxicity", "neural toxicity",
 "drug-induced neurotoxicity", "nervous system toxicity"},
 
 #
 "Carcinogenicity": {"carcinogenicity", "", "carcinogenic", "cancer risk",
 "tumorigenic", "oncogenic", "cancer-causing"},
 
 #
 "Skin Sensitization": {"skin sensitization", " ", "skin allergy", "dermatitis",
 "contact sensitization", "allergic reaction", "hypersensitivity"},
 
 #
 "Eye Irritation": {"eye irritation", "", "ocular irritation", "eye toxicity",
 "ocular toxicity", "eye damage", "corneal irritation"},
 
 #
 "Respiratory": {"respiratory", "", "lung toxicity", "pulmonary toxicity",
 "respiratory toxicity", "breathing problems", "lung damage"},
 
 #
 "SAscore": {"sascore", " ", "synthesizability", "synthetic accessibility",
 "synthesis difficulty", "hard to synthesize", "synthetic challenge"},
 
 # drugsimilar
 "QED": {"qed", "drug similar", "drug-likeness", "drug like", "desirability",
 "drug similarity", "pharmaceutical properties"},
 
 #
 "logS": {"logs", "", "solubility", "aqueous solubility", "water solubility",
 "dissolution", "solubility profile"},
 
 #
 "logP": {"logp", "clogp", "lipophilicity", "hydrophobicity", "partition coefficient",
 "octanol-water partition", "lipid solubility"},
 
 # molecule
 "Molecular Weight (MW)": {"molecular weight (mw)", "molecular weight", "mw", "molecule",
 "molecular mass", "compound weight", "molecule size"},
 
 #
 "Topological Polar Surface Area (TPSA)": {"topological polar surface area (tpsa)", 
 "tpsa", "", "polar surface area",
 "PSA", "polar area", "surface polarity"},
 
 # key
 "HBD (nHD)": {"hbd (nhd)", "hbd", "nhd", "", "hydrogen bond donor", "HBD",
 "hydrogen donors", "donor groups"},
 
 "HBA (nHA)": {"hba (nha)", "hba", "nha", "", "hydrogen bond acceptor", "HBA",
 "hydrogen acceptors", "acceptor groups"},
 
 # key
 "Flexibility": {"flexibility", "", "rotatable bonds", "num rotatable bonds", 
 "nrot", "roam", "key ", "molecular flexibility", "conformational flexibility"},
 
 #
 "BBB": {"bbb", "blood brain barrier", "", "brain barrier", "CNS penetration",
 "blood-brain barrier", "brain access", "CNS access"},
 
 #
 "PPB": {"ppb", "plasma protein binding", " ", "protein binding",
 "plasma binding", "serum protein binding", "albumin binding"},
 
 #
 "Fu": {"fu", " ", "fraction unbound", "unbound fraction", "fu_plasma",
 "fu (plasma)", "fu%", "fu - plasma", "free fraction", "unbound drug"},
 
 #
 "T1/2": {"t1/2", "", "half-life", "elimination half-life", "plasma half-life",
 "clearance half-life", "metabolic half-life"},
 
 # clear
 "CL_plasma": {"cl_plasma", "cl plasma", "cl<sub>plasma</sub>", " clear",
 "plasma clearance", "systemic clearance", "total clearance", "CL"},
 
 #
 "HIA": {"hia", "human intestinal absorption", "", "intestinal absorption",
 "oral absorption", "GI absorption", "gastrointestinal absorption"},
 
 #
 "Rat Oral Acute Toxicity": {"rat oral acute toxicity", " ",
 "acute oral toxicity", "LD50", "lethal dose", "acute lethality"},
 
 #
 "Aquatic Toxicity Rule": {"aquatic toxicity rule", " rule", "water toxicity",
 "aquatic life toxicity", "environmental toxicity"},
 
 #
 "BCF": {"bcf", "", "bioconcentration factor", "bioaccumulation",
 "tissue accumulation", "biomagnification"},
 
 #
 "IGC50": {"igc50", "inhibitory concentration", "growth inhibition", "cell viability"},
 "LC50DM": {"lc50dm", "lethal concentration", "mortality concentration"},
 "LC50FM": {"lc50fm", "fish mortality", "aquatic lethality"},
 
 # rule
 "Lipinski Rule": {"lipinski rule", "lipinski rule", "rule of five", "drug-like properties",
 "Lipinski's rule", "molecular properties rule"},
 
 "GoldenTriangle": {"goldentriangle", "rule", "golden triangle", "ADMET triangle",
 "property triangle", "drug optimization triangle"},
 
 # scoring
 "Fsp3": {"fsp3", "fsp³", "fsp 3", "fsp<sup>3</sup>", "fraction sp3", "sp3 carbon fraction",
 "saturation", "molecular saturation"},
 
 "NPscore": {"npscore", "np ", "natural product score", "natural product likeness",
 "NP likeness", "natural product similarity"},
 
 #
 "nRing": {"nring", "", "number of rings", "ring count", "aromatic rings",
 "cyclic structures", "ring systems"},
 
 "nHet": {"nhet", "", "heteroatoms", "heteroatom count", "non-carbon atoms",
 "hetero atoms", "heteroatom number"},
 
 "nRig": {"nrig", " ", "rigid bonds", "rigid atom count", "rigidity"},
 
 #
 "Stereo Centers": {"stereo centers", " ", "chiral centers", "stereocenters",
 "asymmetric centers", "chirality", "stereochemistry"},
 
 #
 "Melting point": {"melting point", "", "mp", "fusion point", "liquefaction point"},
 "Boiling point": {"boiling point", "", "bp", "vaporization point", "evaporation point"},
 "Density": {"density", "", "specific gravity", "mass density", "bulk density"},
 
 #
 "F20%": {"f20%", "fraction at 20%", "20% fraction"},
 "F30%": {"f30%", "fraction at 30%", "30% fraction"},
 "F50%": {"f50%", "fraction at 50%", "50% fraction"},
 
 # value
 "logD7.4": {"logd7.4", "logd 7.4", "logd_7_4", "distribution coefficient", "logD at pH 7.4"},
 
 # rule
 "MCE-18": {"mce-18", "MCE rule", "molecular complexity", "complexity rule"},
 "GASA": {"gasa", "GASA score", "synthetic accessibility", "synthesis difficulty"},
 "FDAMDD": {"fdamdd", "fda max", "fda mdd", "FDA maximum daily dose", "maximum dose"},
 "IG C50": {"ig c50", "igc50", "inhibitory concentration", "growth inhibition"},
 
 #
 "BCRP inhibitor": {"bcrp inhibitor", "bcrp ", "bcrp", "breast cancer resistance protein"},
 "FLuc inhibitors": {"fluc inhibitors", " ", "firefly luciferase inhibitors"},
 
 #
 "CYP1A2 substrate": {"cyp1a2 substrate", "cyp1a2 ", "1A2 substrate", "CYP1A2 metabolism"},
 "CYP2B6 substrate": {"cyp2b6 substrate", "cyp2b6 ", "2B6 substrate", "CYP2B6 metabolism"},
 "CYP2C19 substrate": {"cyp2c19 substrate", "cyp2c19 ", "2C19 substrate", "CYP2C19 metabolism"},
 "CYP2C9 substrate": {"cyp2c9 substrate", "cyp2c9 ", "2C9 substrate", "CYP2C9 metabolism"},
 "CYP2D6 substrate": {"cyp2d6 substrate", "cyp2d6 ", "2D6 substrate", "CYP2D6 metabolism"},
 "CYP3A4 substrate": {"cyp3a4 substrate", "cyp3a4 ", "3A4 substrate", "CYP3A4 metabolism"},
 
 #
 "CYP1A2 inhibitor": {
 "cyp1a2 inhibitor", "cyp1A2 inhibitor", "CYP1a2 inhibitor", "CYP1A2 inhibitor",
 "cyp1a2 ", "CYP1a2 ", "1A2 inhibitor", "1a2 inhibitor"
 },
 "CYP2B6 inhibitor": {"cyp2b6 inhibitor", "cyp2b6 ", "2B6 inhibitor"},
 "CYP2C19 inhibitor": {
 "cyp2c19 inhibitor", "cyp2C19 inhibitor", "CYP2c19 inhibitor", "CYP2C19 inhibitor",
 "cyp2c19 ", "CYP2c19 ", "2C19 inhibitor", "2c19 inhibitor"
 },
 "CYP2C8 inhibitor": {"cyp2c8 inhibitor", "cyp2c8 ", "2C8 inhibitor"},
 "CYP2C9 inhibitor": {
 "cyp2c9 inhibitor", "cyp2C9 inhibitor", "CYP2c9 inhibitor", "CYP2C9 inhibitor",
 "cyp2c9 ", "CYP2c9 ", "2C9 inhibitor", "2c9 inhibitor"
 },
 "CYP2D6 inhibitor": {"cyp2d6 inhibitor", "cyp2d6 ", " cyp2d6", "cyp2d6/", "2D6 inhibitor"},
 "CYP3A4 inhibitor": {
 "cyp3a4 inhibitor", "cyp3A4 inhibitor", "CYP3a4 inhibitor", "CYP3A4 inhibitor",
 "cyp3a4 ", "CYP3a4 ", "3A4 inhibitor", "3a4 inhibitor"
 },
 
 #
 "Toxicity: Eye Corrosion": {"toxicity: eye corrosion", "", "ocular corrosion", "eye damage"},
 "Toxicity: Eye Irritation": {"toxicity: eye irritation", "", "ocular irritation", "eye irritation"},
 "Toxicity: Human Hepatotoxicity": {"toxicity: human hepatotoxicity", "class ", "human liver toxicity"},
 "Toxicity: Drug-induced Nephrotoxicity": {"toxicity: drug-induced nephrotoxicity", "drug ", 
 "nephrotoxicity", "kidney toxicity", "renal toxicity"},
 "Toxicity: Drug-induced Neurotoxicity": {"toxicity: drug-induced neurotoxicity", "drug ",
 "neurotoxicity", "CNS toxicity", "brain toxicity"},
 "Toxicity: Ototoxicity": {"toxicity: ototoxicity", "", "hearing toxicity", "auditory toxicity"},
 "Toxicity: Hematotoxicity": {"toxicity: hematotoxicity", " ", "blood toxicity", "hematologic toxicity"},
 "Toxicity: Genotoxicity": {"toxicity: genotoxicity", " ", "genetic toxicity", "DNA toxicity"},
 "Toxicity: Carcinogenicity": {"toxicity: carcinogenicity", "", "cancer toxicity", "tumorigenic"},
 "Toxicity: Skin Sensitization": {"toxicity: skin sensitization", " ", "skin allergy", "dermatitis"},
 "Toxicity: DILI": {"toxicity: dili", "drug ", "drug-induced liver injury"},
 "Toxicity: AMES Toxicity": {"toxicity: ames toxicity", "ames ", "mutagenicity", "AMES test"},
 "Toxicity: Rats Oral Acute Toxicity": {"toxicity: rats oral acute toxicity", " ", 
 "acute oral toxicity", "LD50"},
 "Toxicity: FDAMDD": {"toxicity: fdamdd", "fda mdd", "FDA maximum daily dose", "maximum dose"},
 "Toxicity: RPMI-8226 Immunitoxicity": {"toxicity: rpmi-8226 immunitoxicity", "rpmi-8226 ",
 "immune toxicity", "immunotoxicity"},
 "Toxicity: A549 Cytotoxicity": {"toxicity: a549 cytotoxicity", "a549 ", "lung cell toxicity"},
 "Toxicity: Hek293 Cytotoxicity": {"toxicity: hek293 cytotoxicity", "hek293 ", "kidney cell toxicity"},
 "Toxicity: BCF": {"toxicity: bcf", "bioconcentration", "bioaccumulation"},
 "Toxicity: IGC50": {"toxicity: igc50", "inhibitory concentration", "growth inhibition"},
 "Toxicity: LC50DM": {"toxicity: lc50dm", "lethal concentration", "mortality"},
 "Toxicity: LC50FM": {"toxicity: lc50fm", "fish mortality", "aquatic lethality"},
 
 # rule
 "BMS Rule": {"bms rule", "bms rule", "Bristol-Myers Squibb rule", "BMS criteria"},
 "GSK Rule": {"gsk rule", "gsk rule", "GlaxoSmithKline rule", "GSK criteria"},
 "Pfizer Rule": {"pfizer rule", "pfizer rule", "Pfizer criteria", "Pfizer guidelines"},
 
 # rule
 "Alarm_NMR Rule": {"alarm_nmr rule", " nmr rule", "NMR alarm", "NMR alerts"},
 "Acute Toxicity Rule": {"acute toxicity rule", " rule", "acute toxicity alerts"},
 "Chelating Rule": {"chelating rule", " rule", "metal chelation", "chelating agents"},
 "Genotoxic Carcinogenicity Mutagenicity Rule": {"genotoxic carcinogenicity mutagenicity rule", 
 " rule", "genotoxicity alerts"},
 "NonGenotoxic Carcinogenicity Rule": {"nongenotoxic carcinogenicity rule", " rule",
 "non-genotoxic alerts"},
 "Skin Sensitization Rule": {"skin sensitization rule", " rule", 
 "skin sensitization alerts"},
 
 #
 "NR-AR": {"nr-ar", "androgen receptor", "AR receptor", "androgen binding"},
 "NR-AR-LBD": {"nr-ar-lbd", "androgen receptor ligand binding", "AR LBD"},
 "NR-AhR": {"nr-ahr", "aryl hydrocarbon receptor", "AhR receptor", "dioxin receptor"},
 "NR-Aromatase": {"nr-aromatase", "nr-", "aromatase enzyme", "estrogen synthesis"},
 "NR-ER": {"nr-er", "estrogen receptor", "ER receptor", "estrogen binding"},
 "NR-ER-LBD": {"nr-er-lbd", "estrogen receptor ligand binding", "ER LBD"},
 "NR-PPAR-gamma": {"nr-ppar-gamma", "nr-ppar-γ", "PPAR gamma", "peroxisome proliferator"},
 
 # reaction
 "SR-ARE": {"sr-are", "antioxidant response element", "ARE pathway", "oxidative stress"},
 "SR-ATAD5": {"sr-atad5", "ATAD5 stress response", "DNA replication stress"},
 "SR-HSE": {"sr-hse", "heat shock element", "heat shock response", "thermal stress"},
 "SR-MMP": {"sr-mmp", "mitochondrial membrane potential", "MMP stress", "mitochondrial stress"},
 "SR-p53": {"sr-p53", "p53 stress response", "tumor suppressor", "DNA damage response"},
 
 #
 "A549 Cytotoxicity": {"a549 cytotoxicity", "a549 ", "lung cell toxicity",
 "A549 cell line", "pulmonary cytotoxicity"},
 "Hek293 Cytotoxicity": {"hek293 cytotoxicity", "hek293 ", "kidney cell toxicity",
 "HEK293 cell line", "renal cytotoxicity"},
 
 #
 "RPMI-8226 Immunitoxicity": {"rpmi-8226 immunitoxicity", "rpmi-8226 ",
 "RPMI8226 toxicity", "myeloma cell toxicity", "immune toxicity"},
 
 #
 "NonBiodegradable": {"nonbiodegradable", " ", "non-biodegradable",
 "persistent", "environmental persistence", "biodegradation resistance"},
 "PAINS": {"pains", "pan-assay interference compounds", "assay interference",
 "false positives", "promiscuous binders"},
 "Promiscuous compounds": {"promiscuous compounds", " ", "promiscuous binders",
 "non-specific binding", "off-target binding", "polypharmacology"},
 "Reactive compounds": {"reactive compounds", "", "reactive metabolites",
 "electrophilic compounds", "reactive intermediates"},
 "Green fluorescence": {"green fluorescence", "", "fluorescent", "fluorescence",
 "autofluorescence", "background fluorescence"},
 "PAMPA": {"pampa", "parallel artificial membrane permeability", "artificial membrane",
 "membrane model", "permeability assay"},
 "MDCK Permeability": {"mdck permeability", "mdck", "Madin-Darby canine kidney",
 "canine kidney permeability", "MDCK model"},
 "CYP1A2 substrate": {
 "cyp1a2 substrate", "cyp1A2 substrate", "CYP1a2 substrate", "CYP1A2 substrate",
 "cyp1a2 ", "CYP1a2 ", "1A2 substrate", "1a2 substrate"
 },
 "CYP2C19 substrate": {
 "cyp2c19 substrate", "cyp2C19 substrate", "CYP2c19 substrate", "CYP2C19 substrate",
 "cyp2c19 ", "CYP2c19 ", "2C19 substrate", "2c19 substrate"
 },
})


# —— buildmapping() -> —— #
REV_ALIAS: Dict[str, str] = {}
for canon, synonyms in ALIAS.items():
 for s in synonyms | {canon}:
 key = s.lower().strip()
 REV_ALIAS[key] = canon

# === [PATCH 1] mapping ===
CANONICAL_MERGE: Dict[str, str] = {
 "nRot": "Flexibility",
 "Roam (nRot)": "Flexibility",
 "nHD": "HBD (nHD)",
 "nHA": "HBA (nHA)",
}
# —— "Drug induced X" Toxicity —— #
CANONICAL_MERGE.update({
 "Drug induced Neurotoxicity": "Toxicity: Drug-induced Neurotoxicity",
 "Drug-induced Neurotoxicity": "Toxicity: Drug-induced Neurotoxicity",
 "Drug induced Nephrotoxicity": "Toxicity: Drug-induced Nephrotoxicity",
 "Drug-induced Nephrotoxicity": "Toxicity: Drug-induced Nephrotoxicity",
 "Drug induced Ototoxicity": "Toxicity: Ototoxicity",
 "Drug-induced Ototoxicity": "Toxicity: Ototoxicity",
 "Drug induced Hematotoxicity": "Toxicity: Hematotoxicity",
 "Drug-induced Hematotoxicity": "Toxicity: Hematotoxicity",
})
# —— allformatch —— #
import string
def _flatten_alnum(s: str) -> str:
 return re.sub(r"[^0-9a-zA-Z]+", "", s or "").lower()

REV_ALIAS_FLAT: Dict[str, str] = {}
for k_low, canon in REV_ALIAS.items():
 REV_ALIAS_FLAT[_flatten_alnum(k_low)] = canon

def normalize_key(k: str) -> str:
 if not k:
 return ""
 s = str(k)
 s = re.sub(r"<[^>]+>", "", s) # HTML
 s = s.replace("", "(").replace("", ")")
 raw_low = s.lower().strip()

 if raw_low in REV_ALIAS:
 s = REV_ALIAS[raw_low]
 else:
 s2 = re.sub(r"(?<=[A-Za-z])(?=[A-Z][a-z])", " ", s)
 s2 = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", s2)
 s2 = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", s2)
 low2 = s2.lower().strip()
 if low2 in REV_ALIAS:
 s = REV_ALIAS[low2]
 else:
 s3 = re.sub(r"[_\-]+", " ", s2)
 s3 = re.sub(r"\s+", " ", s3).strip()
 low3 = s3.lower()
 s = REV_ALIAS.get(low3, s3)

 s = CANONICAL_MERGE.get(s, s)
 flat = _flatten_alnum(s)
 s = REV_ALIAS_FLAT.get(flat, s)
 return s

def normalize_props(props: Dict[str, Any]) -> Dict[str, Any]:
 out: Dict[str, Any] = {}
 for k, v in props.items():
 nk = normalize_key(k)
 out[nk] = v
 return out

# —— key —— #
UP_KW = ("", "", "increase", "up", "", "", "enhance", "improve", "boost", "raise")
DOWN_KW = ["", "", "decrease", "down", "", "", "reduce", "lower", "mitigate", "cut", "less", "drop"]

############################################################
# 1) metric/target
############################################################

RANGE_TARGETS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
 "logP": (1.0, 3.0),
 "Topological Polar Surface Area (TPSA)": (20.0, 130.0), # Å²
 "Molecular Weight (MW)": (150.0, 500.0), # Da
}

MORE_IS_BETTER_BASE: Set[str] = {
 "HLM Stability",
 "logS",
 "logD7.4",
 "Flexibility",
 "Fsp3",
 "QED",
 "NPscore",
 "Caco-2 Permeability",
 "MDCK Permeability",
 "PAMPA",
 "Fu",
}

def build_less_is_better(all_features: List[str]) -> Set[str]:
 s: Set[str] = set()
 for f in all_features:
 fn = normalize_key(f)
 if fn.lower().endswith("inhibitor"):
 s.add(fn)
 if fn.lower().startswith("toxicity:"):
 s.add(fn)
 s.update({normalize_key(x) for x in [
 "Human Hepatotoxicity",
 "Genotoxicity",
 "Carcinogenicity",
 "Eye Irritation",
 "Eye Corrosion",
 "Respiratory",
 "Skin Sensitization",
 "DILI",
 "AMES Toxicity",
 "PPB"
 ]})
 return s

ALL_FEATURES_CANON = sorted({normalize_key(f) for f in FEATURES})

# ADMETfeatureforcompute
CORE_FEATURES = {
 "Caco-2 Permeability", "F50%", "CYP3A4 inhibitor", "CYP2D6 inhibitor",
 "Pgp substrate", "hERG Blockers", "DILI", "Human Hepatotoxicity",
 "AMES Toxicity", "Genotoxicity", "Toxicity: Drug-induced Neurotoxicity",
 "QED", "SAScore", "GASA", "Lipinski Rule", "HLM Stability"
}

LESS_IS_BETTER: Set[str] = build_less_is_better(FEATURES)
MORE_IS_BETTER: Set[str] = set(
 normalize_key(f) for f in MORE_IS_BETTER_BASE
 if normalize_key(f) not in RANGE_TARGETS
)

def is_more_is_better(feat: str) -> Optional[bool]:
 if feat in MORE_IS_BETTER:
 return True
 if feat in LESS_IS_BETTER:
 return False
 if feat in RANGE_TARGETS:
 return None
 return None

############################################################
# 2) threshold
############################################################

def load_thresholds() -> Dict[str, float]:
 thr: Dict[str, float] = {}
 for _cat, feat_dict in checks.items():
 for raw_feat, (flag, threshold, _msg) in feat_dict.items():
 try:
 t = float(threshold)
 except Exception:
 continue
 canon = normalize_key(raw_feat)
 thr[canon] = t
 return thr

THR_MAP = load_thresholds()
# rewardparameter
BASE_IMPROVEMENT_BONUS = 2.0 # 
RELATIVE_IMPROVEMENT_WEIGHT = 1.5 # weightbased on
THR_BONUS = 2.5 # thresholdthreshold
RANGE_BONUS = 3.5 # optimization
MAX_REWARD_PER_FEAT = 2.0 # eachfeaturemaxvalue
# parameter
BASE_DEGRADATION_PENALTY = -1.5 # 
RELATIVE_DEGRADATION_WEIGHT = -1.2 # weightbased on
THR_PENALTY = -2.0 # threshold
MIN_PENALTY_PER_FEAT = -2.0 # eachfeaturemaxvalue

# featureweight - according tofeaturedifferentweight
FEATURE_WEIGHTS = {
 # keyfeature- weight3.0
 "Genotoxicity": 3.0,
 "DILI": 3.0,
 "Carcinogenicity": 3.0,
 "AMES Toxicity": 2.5,
 "Human Hepatotoxicity": 2.5,
 
 # keyfeaturePK- weight2.0
 "hERG Blockers": 2.0,
 "hERG Blockers (10 um)": 2.0,
 "Toxicity: Drug-induced Neurotoxicity": 2.0,
 "Toxicity: Drug-induced Nephrotoxicity": 2.0,
 "Hematotoxicity": 1.8,
 
 # keyfeature- weight1.5
 "Caco-2 Permeability": 1.5,
 "HIA": 1.5,
 "BBB": 1.5,
 "logS": 1.5,
 "PPB": 1.3,
 "Fu": 1.3,
 
 # feature- weight1.0
 "CYP3A4 inhibitor": 1.0,
 "CYP2D6 inhibitor": 1.0,
 "CYP2C9 inhibitor": 1.0,
 "CYP2C19 inhibitor": 1.0,
 "HLM Stability": 1.0,
 "Pgp substrate": 1.0,
 "Pgp inhibitor": 1.0,
 "QED": 1.0,
 "logP": 1.0,
 "logD7.4": 1.0,
 
 # defaultweight1.0allfeature
}

def get_feature_weight(feat: str) -> float:
 """getfeatureweightdefault1.0"""
 return FEATURE_WEIGHTS.get(feat, 1.0)

DELTA_WEIGHT = 0.3

############################################################
# 3) valueparse
############################################################

_SCI_RE = re.compile(
 r'(?P<sign>[<>]=?|~)?\s*(?P<val>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(?P<unit>[munp]M|μM|uM|nM|mM|%)?',
 re.IGNORECASE
)

def _convert_unit(val: float, unit: Optional[str]) -> float:
 if not unit:
 return val
 u = unit.strip().lower()
 if u == "mm":
 return val * 1000.0
 if u in ("um", "μm"):
 return val
 if u == "nm":
 return val / 1000.0
 if u == "pm":
 return val / 1_000_000.0
 if u == "%":
 return val
 return val

def parse_numeric_value(raw: str) -> Optional[float]:
 if raw is None:
 return None
 s = str(raw)
 m = _SCI_RE.search(s)
 if not m:
 return None
 try:
 val = float(m.group("val"))
 except Exception:
 return None
 unit = m.group("unit")
 val = _convert_unit(val, unit)
 return val

def map_admet_value(feature: str, raw: Any) -> Optional[float]:
 """
 rule
 - parsevaluereturns μM
 - +/-" → value"/
 ""judge reward_for_improvement()
 LESS_IS_BETTER → p < o 
 """
 raw_s = "" if raw is None else str(raw).strip()

 # 1) value
 num = parse_numeric_value(raw_s)
 if num is not None:
 return num

 # 2) value +/- score = pluses - minuses
 pluses = raw_s.count("+")
 minuses = raw_s.count("-")
 if pluses == 0 and minuses == 0:
 return None

 score = float(pluses - minuses)
 return score


def extract_numeric(props: Dict[str, Any]) -> Dict[str, float]:
 numeric: Dict[str, float] = {}
 for feat, raw in props.items():
 v = map_admet_value(feat, raw)
 if v is not None:
 numeric[feat] = float(v)
 return numeric

############################################################
# 4) Reasoning parsebonus 
############################################################
def _alias_hits(txt: str, alias: str):
 a = alias.lower()
 if re.fullmatch(r"[a-z0-9][a-z0-9 _\-]*[a-z0-9]", a):
 return [m.span() for m in re.finditer(rf"\b{re.escape(a)}\b", txt)]
 i = txt.find(a)
 return [(i, i + len(a))] if i >= 0 else []

def parse_reasoning_enhanced(reasoning: str) -> List[Tuple[str, str]]:
 """
 inferenceparsefunctiondescription
 """
 txt = (reasoning or "").lower()
 pairs: List[Tuple[str, str]] = []
 WINDOW = 100
 
 # 1. recognition - recognitioninference
 # useADMETdatafeature
 problem_mappings = {
 # inferencerecognitionfeature
 "low caco-2 permeability": ("Caco-2 Permeability", "down"),
 "poor permeability": ("Caco-2 Permeability", "down"),
 "permeability issues": ("Caco-2 Permeability", "down"),
 "permeability": ("Caco-2 Permeability", "down"),
 "low solubility": ("logS", "down"),
 "poor solubility": ("logS", "down"),
 "solubility issues": ("logS", "down"),
 "solubility": ("logS", "down"),
 "high clearance": ("CL_plasma", "up"),
 "rapid clearance": ("CL_plasma", "up"),
 "clearance issues": ("CL_plasma", "up"),
 "clearance": ("CL_plasma", "up"),
 "cyp1a2 inhibition": ("CYP1A2 inhibitor", "up"),
 "cyp1a2 inhibitor": ("CYP1A2 inhibitor", "up"),
 "cyp3a4 inhibition": ("CYP3A4 inhibitor", "up"),
 "cyp3a4 inhibitor": ("CYP3A4 inhibitor", "up"),
 "cyp2c19 inhibition": ("CYP2C19 inhibitor", "up"),
 "cyp2c19 inhibitor": ("CYP2C19 inhibitor", "up"),
 "cyp2c9 inhibition": ("CYP2C9 inhibitor", "up"),
 "cyp2c9 inhibitor": ("CYP2C9 inhibitor", "up"),
 "cyp450": ("CYP3A4 inhibitor", "up"),
 "cyp": ("CYP3A4 inhibitor", "up"),
 "herg blockage": ("hERG", "up"),
 "herg inhibition": ("hERG", "up"),
 "cardiac issues": ("hERG", "up"),
 "genotoxicity": ("Genotoxicity", "up"),
 "mutagenicity": ("Genotoxicity", "up"),
 "dna damage": ("Genotoxicity", "up"),
 "hepatotoxicity": ("DILI", "up"),
 "liver toxicity": ("DILI", "up"),
 "liver injury": ("DILI", "up"),
 "dili": ("DILI", "up"),
 "neurotoxicity": ("Toxicity: Drug-induced Neurotoxicity", "up"),
 "cns toxicity": ("Toxicity: Drug-induced Neurotoxicity", "up"),
 "brain toxicity": ("Toxicity: Drug-induced Neurotoxicity", "up"),
 "ames toxicity": ("AMES Toxicity", "up"),
 "ames test": ("AMES Toxicity", "up"),
 "mutagenic": ("AMES Toxicity", "up"),
 "carcinogenicity": ("Carcinogenicity", "up"),
 "cancer risk": ("Carcinogenicity", "up"),
 "skin sensitization": ("Skin Sensitisation", "up"),
 "allergic reaction": ("Skin Sensitisation", "up"),
 "eye irritation": ("Eye Irritation", "up"),
 "ocular toxicity": ("Eye Irritation", "up"),
 "respiratory toxicity": ("Respiratory", "up"),
 "lung toxicity": ("Respiratory", "up"),
 "metabolic instability": ("HLM Stability", "down"),
 "metabolism issues": ("HLM Stability", "down"),
 "metabolism": ("HLM Stability", "down"),
 "synthesis difficulty": ("SAScore", "up"),
 "hard to synthesize": ("SAScore", "up"),
 "drug-likeness": ("QED", "down"),
 "desirability": ("QED", "down"),
 # featuredescription
 "dili risk": ("DILI", "up"),
 "liver injury risk": ("DILI", "up"),
 "hepatotoxicity risk": ("DILI", "up"),
 "genotoxicity risk": ("Genotoxicity", "up"),
 "mutagenicity risk": ("Genotoxicity", "up"),
 "ames positive": ("AMES Toxicity", "up"),
 "neurotoxicity risk": ("Toxicity: Drug-induced Neurotoxicity", "up"),
 "cns toxicity risk": ("Toxicity: Drug-induced Neurotoxicity", "up"),
 "human hepatotoxicity": ("Human Hepatotoxicity", "up"),
 "liver toxicity": ("Human Hepatotoxicity", "up"),
 "hepatotoxicity risk": ("Human Hepatotoxicity", "up"),
 "liver injury risk": ("Human Hepatotoxicity", "up"),
 # featuredescription
 "bioactivation": ("DILI", "up"),
 "metabolic soft spot": ("DILI", "up"),
 "metabolically stable": ("HLM Stability", "up"),
 "aromaticity": ("logP", "up"),
 "planarity": ("logP", "up"),
 "polarity": ("logS", "up"),
 "hydrogen bond": ("logS", "up"),
 "hydrogen bonding": ("logS", "up"),
 # recognition
 "": ("DILI", "up"),
 "": ("DILI", "up"),
 "drug": ("DILI", "up"),
 "drug": ("DILI", "up"),
 "hepatotoxic": ("DILI", "up"),
 "liver damage": ("DILI", "up"),
 "hepatic injury": ("DILI", "up"),
 "hepatic toxicity": ("DILI", "up"),
 "liver safety": ("DILI", "up"),
 "hepatotoxic risk": ("DILI", "up"),
 "liver injury risk": ("DILI", "up"),
 "hepatotoxicity concern": ("DILI", "up"),
 "liver toxicity concern": ("DILI", "up"),
 "hepatotoxic potential": ("DILI", "up"),
 "liver toxicity potential": ("DILI", "up"),
 #
 "": ("HLM Stability", "down"),
 "": ("HLM Stability", "down"),
 "clear": ("HLM Stability", "down"),
 "metabolic clearance": ("HLM Stability", "down"),
 "metabolic stability": ("HLM Stability", "up"),
 "metabolism stability": ("HLM Stability", "up"),
 "hepatic metabolism": ("HLM Stability", "down"),
 "liver metabolism": ("HLM Stability", "down"),
 "oxidative metabolism": ("HLM Stability", "down"),
 "metabolic activation": ("HLM Stability", "down"),
 #
 "": ("Caco-2 Permeability", "down"),
 "": ("Caco-2 Permeability", "down"),
 "": ("Caco-2 Permeability", "down"),
 "intestinal permeability": ("Caco-2 Permeability", "down"),
 "membrane permeability": ("Caco-2 Permeability", "down"),
 "passive permeability": ("Caco-2 Permeability", "down"),
 "absorption": ("Caco-2 Permeability", "down"),
 "bioavailability": ("Caco-2 Permeability", "down"),
 # molecule
 "molecular weight": ("Molecular Weight (MW)", "down"),
 "molecular mass": ("Molecular Weight (MW)", "down"),
 "high molecular weight": ("Molecular Weight (MW)", "down"),
 "large molecule": ("Molecular Weight (MW)", "down"),
 "heavy molecule": ("Molecular Weight (MW)", "down"),
 # key
 "hydrogen bond donors": ("nHD", "down"),
 "hbd": ("nHD", "down"),
 "hydrogen bond acceptors": ("nHA", "down"),
 "hba": ("nHA", "down"),
 # logP
 "lipophilicity": ("logP", "down"),
 "hydrophobicity": ("logP", "down"),
 "high logp": ("logP", "down"),
 "low logp": ("logP", "up"),
 # key
 "molecule": ("Molecular Weight (MW)", "down"),
 "key": ("nHD", "down"),
 "key": ("nHA", "down"),
 "": ("logP", "down"),
 "": ("logP", "down"),
 "": ("logS", "up"),
 "": ("Caco-2 Permeability", "up"),
 "": ("HLM Stability", "up"),
 "": ("HLM Stability", "down"),
 }
 
 # checkinference
 for problem_text, (feature, direction) in problem_mappings.items():
 if problem_text in txt:
 pairs.append((feature, direction))
 
 # process
 context_keywords = ["liabilities", "concerns", "problems", "issues", "need to be addressed", 
 "key liabilities", "main concerns", "several liabilities"]
 for keyword in context_keywords:
 if keyword in txt:
 # keyfeaturedescription
 keyword_pos = txt.find(keyword)
 context_window = txt[max(0, keyword_pos-200):keyword_pos+200]
 
 # feature
 for problem_text, (feature, direction) in problem_mappings.items():
 if problem_text in context_window:
 pairs.append((feature, direction))
 
 # originalfeatureforF1compute
 unique_pairs = []
 seen_features = set()
 for feature, direction in pairs:
 if feature not in seen_features:
 unique_pairs.append((feature, direction))
 seen_features.add(feature)
 
 return unique_pairs

# replaceparse_reasoningfunction
def parse_reasoning(reasoning: str) -> List[Tuple[str, str]]:
 return parse_reasoning_enhanced(reasoning)

############################################################
# 5) threshold/
############################################################

def _interval_distance_to_band(x: float, lo: Optional[float], hi: Optional[float]) -> float:
 if lo is not None and x < lo:
 return lo - x
 if hi is not None and x > hi:
 return x - hi
 return 0.0

def reward_for_improvement(o: float, p: float, feat: str) -> float:
 """
 newlayer+ + featureweight
 1. 
 2. based on
 3. thresholdthreshold
 4. optimization
 
 
 5. 
 6. based on
 7. threshold
 
 featureweight
 8. according tofeaturefeatureweight
 """
 score = 0.0
 
 # getfeatureweight
 weight = get_feature_weight(feat)
 
 # 1) 
 if feat in RANGE_TARGETS:
 lo, hi = RANGE_TARGETS[feat]
 dist_o = _interval_distance_to_band(o, lo, hi)
 dist_p = _interval_distance_to_band(p, lo, hi)
 delta_eff = (dist_o - dist_p)
 
 # optimization
 if dist_o > 0 and dist_p == 0:
 score += RANGE_BONUS
 
 #
 if dist_o == 0 and dist_p > 0:
 score += -RANGE_BONUS * 0.8 # 
 
 # /
 if delta_eff > 0:
 score += RELATIVE_IMPROVEMENT_WEIGHT * min(delta_eff, 2.0)
 elif delta_eff < 0:
 #
 score += RELATIVE_DEGRADATION_WEIGHT * max(delta_eff, -2.0)
 
 # featureweightreturns
 weighted_score = score * weight
 return max(min(weighted_score, MAX_REWARD_PER_FEAT * weight), MIN_PENALTY_PER_FEAT * weight)
 
 # 2) 
 direction_more = is_more_is_better(feat)
 if direction_more is None:
 return 0.0
 
 delta_eff = (p - o) if direction_more else (o - p)
 
 # vs 
 if delta_eff > 0:
 score += BASE_IMPROVEMENT_BONUS
 elif delta_eff < 0:
 # - 
 abs_change = abs(delta_eff)
 relative_change = abs_change / max(abs(o), 0.1)
 
 if relative_change < 0.05:
 # <5%: 
 score += BASE_DEGRADATION_PENALTY * 0.2
 elif relative_change < 0.15:
 # 5-15%: 
 score += BASE_DEGRADATION_PENALTY * 0.5
 else:
 # >15%: 
 score += BASE_DEGRADATION_PENALTY
 
 # threshold/
 if feat in THR_MAP:
 T = THR_MAP[feat]
 if direction_more:
 if o < T <= p:
 score += THR_BONUS # threshold
 elif o >= T > p:
 score += THR_PENALTY # threshold
 else:
 if o > T >= p:
 score += THR_BONUS # threshold
 elif o <= T < p:
 score += THR_PENALTY # threshold
 
 # /
 if delta_eff > 0:
 # compute
 relative_improvement = delta_eff / max(abs(o), 0.1) # 
 score += RELATIVE_IMPROVEMENT_WEIGHT * min(relative_improvement, 2.0)
 elif delta_eff < 0:
 # compute
 relative_degradation = delta_eff / max(abs(o), 0.1)
 score += RELATIVE_DEGRADATION_WEIGHT * max(relative_degradation, -2.0)
 
 # featureweightreturns
 weighted_score = score * weight
 return max(min(weighted_score, MAX_REWARD_PER_FEAT * weight), MIN_PENALTY_PER_FEAT * weight)


############################################################
# 6) featurerecognition
############################################################

def get_bad_features(ro: Dict[str, float]) -> Set[str]:
 bad = set()
 for _cat, feat_dict in checks.items():
 for raw_feat, ft in feat_dict.items():
 feat = normalize_key(raw_feat)
 if feat not in ro:
 continue
 if check_item(ft, str(ro[feat])):
 bad.add(feat)
 return bad

############################################################
# 7) 
############################################################
def _smiles_variants(s: str):
 vs = []
 s0 = (s or "").strip()
 if not s0:
 return vs
 vs.append(s0)
 vs.append(re.sub(r"\s+", "", s0))
 vs.append(re.sub(r"[@/\\]", "", s0))
 vs.append(s0.replace("–", "-").replace("—", "-").replace("−", "-"))
 if "." in s0:
 vs.append(s0.split(".", 1)[0])
 vs.append(re.sub(r"\[[^\]]+\]", "", s0))
 vs.append(re.sub(r"\(\)", "", s0))
 out, seen = [], set()
 for v in vs:
 if v and v not in seen:
 seen.add(v); out.append(v)
 return out



# ====== A1) dependency RDKit ======
_ALLOWED = set("=#()[]+-@\\/0123456789%:.ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz*")

def _basic_smiles_sanity(s: str) -> bool:
 if not s or any(ch not in _ALLOWED for ch in s):
 return False
 # //check
 st = 0
 for ch in s:
 if ch == '(':
 st += 1
 elif ch == ')':
 st -= 1
 if st < 0:
 return False
 if st != 0:
 return False
 # consistent%
 # RDKit
 digits = [c for c in s if c.isdigit()]
 if len(digits) % 2 == 1:
 return False
 return True

# ====== A2) use is_valid.py samevalidate ======
try:
 from rdkit import Chem
 from rdkit.Chem import MolToSmiles
 
 def _is_valid_smiles(s: str) -> bool:
 """validateSTRICT + LENIENT is_valid.py consistent"""
 try:
 # STRICT: sanitize
 mol = Chem.MolFromSmiles(s, sanitize=True)
 if mol is not None:
 return True
 except Exception:
 pass
 
 try:
 # LENIENT: skip sanitize
 mol = Chem.MolFromSmiles(s, sanitize=False)
 if mol is not None:
 return True
 except Exception:
 pass
 
 return False
 
except Exception:
 def _is_valid_smiles(s: str) -> bool:
 return _basic_smiles_sanity(s)


# 1) functionsignature+log
def _safe_search(smiles: str, retries: int = 2, sample_idx: Optional[int] = None, role: str = "") -> Dict[str, Any]:
 """
 recordexecute ADMET 
 - sample idx rolelog
 - retry
 """
 s = (smiles or "").strip()
 s = re.sub(r"\s*<\s*end\s*>\s*$", "", s, flags=re.IGNORECASE)
 if not s:
 if sample_idx is not None:
 print(f"[ADMET-PREDICT][ERROR] Sample {sample_idx} ({role}) empty smiles")
 return {}

 backoff = 0.2 # wait
 last_err = ""
 for t in range(retries + 1):
 try:
 r = search_admet(s) or {}
 if isinstance(r, dict) and r:
 return r
 last_err = f"try{t}: empty dict"
 except Exception as e:
 last_err = f"try{t}: {type(e).__name__}: {e}"
 time.sleep(backoff)
 backoff = min(backoff * 2, 1.6) # maxlatency 1.6 

 tag = f"Sample {sample_idx} ({role})" if sample_idx is not None else role
 print(f"[ADMET-PREDICT][ERROR] {tag} failed. smiles='{s[:120]}', reason={last_err}")
 return {}

# 2) compute_sample idx compute_sample idx parameter
class RewardEngine:
 def __init__(self, w_main: float = 1.0, w_bonus: float = 1.0, coverage_weight: float = 0.3):
 """
 initialize
 
 Args:
 w_main: weightbased onADMETprocess
 w_bonus: F1weightbased onfeaturerecognitionpredictionF1
 coverage_weight: weight
 """
 self.w_main = w_main
 self.w_bonus = w_bonus
 self.coverage_weight = coverage_weight # configweight

 def compute_sample_enhanced(
 self,
 orig_smiles: str,
 opt_smiles: str,
 reasoning: str,
 idx: Optional[int] = None
 ) -> Tuple[float, Dict[str, Any]]:
 # sample
 ro_raw = _safe_search(orig_smiles, sample_idx=idx, role="ro")
 rp_raw = _safe_search(opt_smiles, sample_idx=idx, role="rp")

 #
 ro = extract_numeric(normalize_props(ro_raw))
 rp = extract_numeric(normalize_props(rp_raw))

 if isinstance(ro_raw, dict):
 print("[DEBUG] ro_raw sample keys:", list(ro_raw.keys())[:10])
 if isinstance(rp_raw, dict):
 print("[DEBUG] rp_raw sample keys:", list(rp_raw.keys())[:10])

 print("ro keys:", sorted(ro.keys()))
 print("rp keys:", sorted(rp.keys()))
 print("common:", sorted(set(ro.keys()) & set(rp.keys())))

 main_reward = 0.0
 eval_feats: List[str] = []
 
 # 3based onfeatureevaluationstrategy
 # 1. recognitionoriginalmoleculeoptimizationmoleculefeature
 orig_bad_features = get_bad_features(ro)
 opt_bad_features = get_bad_features(rp)
 
 # 2. evaluationfeature
 for feat in orig_bad_features:
 if feat in rp:
 # featureevaluationwhether
 o = ro.get(feat)
 p = rp.get(feat)
 if o is not None and p is not None:
 if (feat in RANGE_TARGETS) or (is_more_is_better(feat) is not None):
 eval_feats.append(feat)
 main_reward += reward_for_improvement(o, p, feat)
 else:
 # featureneedsvalidateADMETwhethersuccess
 if rp: # ifoptimizationmoleculeADMETdatadescriptionfeature
 main_reward += 0.5
 eval_feats.append(f"{feat}_removed")
 else: # ifADMETfail
 main_reward += 0.1
 eval_feats.append(f"{feat}_unknown")
 
 # 3. evaluationfeature
 new_bad_features = opt_bad_features - orig_bad_features
 for feat in new_bad_features:
 main_reward -= 0.3 # feature
 eval_feats.append(f"{feat}_new_bad")
 
 # 4. evaluationfeaturefeature
 other_features = (set(ro.keys()) & set(rp.keys())) - orig_bad_features
 for feat in other_features:
 o = ro.get(feat)
 p = rp.get(feat)
 if o is not None and p is not None:
 if (feat in RANGE_TARGETS) or (is_more_is_better(feat) is not None):
 eval_feats.append(feat)
 main_reward += reward_for_improvement(o, p, feat) * 0.5 # weight

 # strategy - supports + featureweightrange -1 1
 denom_main = max(len(eval_feats), 1)
 
 # computemaxminfeatureweight
 # eachevaluationfeatureuseweightcomputemax/minvalue
 max_possible_reward = sum(MAX_REWARD_PER_FEAT * get_feature_weight(f.replace('_removed', '').replace('_unknown', '').replace('_new_bad', '')) 
 for f in eval_feats)
 min_possible_penalty = sum(MIN_PENALTY_PER_FEAT * get_feature_weight(f.replace('_removed', '').replace('_unknown', '').replace('_new_bad', '')) 
 for f in eval_feats) # 
 
 # -1 1 range
 if main_reward >= 0:
 # 0-1
 if max_possible_reward > 0:
 base_score = main_reward / max_possible_reward
 else:
 base_score = 0.0
 else:
 # -1-0
 if min_possible_penalty < 0:
 base_score = main_reward / abs(min_possible_penalty) # main_rewardmin_possible_penalty
 else:
 base_score = -1.0
 
 # featurefeature
 # usefeaturecomputefeature
 core_eval_feats = [f for f in eval_feats if f in CORE_FEATURES]
 max_possible_feats = len(CORE_FEATURES)
 coverage_ratio = len(core_eval_feats) / max_possible_feats
 coverage_bonus = min(coverage_ratio, 1.0) * 0.1 # weight
 
 # "" - strategy
 improvement_efficiency_bonus = 0.0
 
 # checkwhether reasoning
 has_reasoning = bool(reasoning and reasoning.strip())
 improved_count = 0
 degraded_count = 0
 
 if len(eval_feats) > 0:
 # statisticsfeaturecount
 
 for feat in eval_feats:
 clean_feat = feat.replace('_removed', '').replace('_unknown', '').replace('_new_bad', '')
 if clean_feat in ro and clean_feat in rp:
 o_val = ro[clean_feat]
 p_val = rp[clean_feat]
 if o_val is not None and p_val is not None:
 direction = is_more_is_better(clean_feat)
 if direction is not None:
 delta = (p_val - o_val) if direction else (o_val - p_val)
 if delta > 0:
 improved_count += 1
 elif delta < 0:
 degraded_count += 1
 elif clean_feat in RANGE_TARGETS:
 lo, hi = RANGE_TARGETS[clean_feat]
 dist_o = _interval_distance_to_band(o_val, lo, hi)
 dist_p = _interval_distance_to_band(p_val, lo, hi)
 if dist_o > dist_p:
 improved_count += 1
 elif dist_p > dist_o:
 degraded_count += 1
 
 # computeefficiency / ( + )
 total_changed = improved_count + degraded_count
 if total_changed > 0:
 efficiency = improved_count / total_changed
 # ifefficiency > 50%
 if efficiency > 0.5:
 improvement_efficiency_bonus = (efficiency - 0.5) * 0.2 # +0.1
 
 # norm_main range -1 1
 # NOTEbonus 
 if base_score >= 0:
 norm_main = min(base_score + coverage_bonus + improvement_efficiency_bonus, 1.0) # 1.0
 else:
 norm_main = max(base_score, -1.0) # -1.0

 # reasoning parsefeature
 if has_reasoning:
 feat_dirs = parse_reasoning(reasoning)
 pred_feats = {f for f, _ in feat_dirs}
 else:
 feat_dirs = []
 pred_feats = set()
 
 bad_feats = get_bad_features(ro)
 
 # if reasoningbonus_f1 0
 if not has_reasoning:
 bonus_hits = 0
 norm_bonus = 0.0
 bonus_f1 = 0.0
 # bonuscompute - usematchstrategy
 elif len(bad_feats) == 0:
 # originalmoleculefeature
 bonus_hits = 0
 denom_bonus = max(len(pred_feats), 1)
 norm_bonus = bonus_hits / denom_bonus
 
 if len(pred_feats) == 0:
 bonus_f1 = 1.0 # featurefeature
 else:
 # featurefeature
 bonus_f1 = 0.5
 else:
 # featureuseF1compute
 bonus_hits = sum(1 for f in pred_feats if f in bad_feats)
 denom_bonus = max(len(pred_feats), 1)
 norm_bonus = bonus_hits / denom_bonus
 
 # F1computeusematchstrategy
 eval_domain = bad_feats | pred_feats
 if len(eval_domain) == 0:
 bonus_f1 = 0.0
 else:
 y_true = [1 if f in bad_feats else 0 for f in eval_domain]
 y_pred = [1 if f in pred_feats else 0 for f in eval_domain]
 
 # fixprocess
 if len(set(y_true)) < 2 and len(set(y_pred)) < 2:
 # ify_truey_predvalue
 bonus_f1 = 1.0 if y_true == y_pred else 0.0
 elif len(set(y_true)) < 2:
 # ify_truevalue01y_predvalue
 if 1 in y_true: # y_true1
 # computerecall
 recall = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1) / len(y_true)
 bonus_f1 = recall
 else: # y_true0
 # compute1-
 fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
 specificity = 1.0 - (fp / len(y_true))
 bonus_f1 = max(specificity, 0.0)
 elif len(set(y_pred)) < 2:
 # ify_predvalue01y_truevalue
 if 1 in y_pred: # y_pred1
 # compute
 precision = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1) / len(y_pred)
 bonus_f1 = precision
 else: # y_pred0
 # compute1-
 fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
 specificity = 1.0 - (fn / len(y_true))
 bonus_f1 = max(specificity, 0.0)
 else:
 # useF1compute
 tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
 fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
 fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
 
 # computerecall
 precision = tp / max(tp + fp, 1)
 recall = tp / max(tp + fn, 1)
 
 if precision + recall == 0:
 bonus_f1 = 0.0
 else:
 # useF2recallweightevaluation
 bonus_f1 = (1 + 4) * (precision * recall) / (4 * precision + recall)
 
 # ifrecallcan
 if precision < 0.3 and recall > 0.5:
 bonus_f1 = max(bonus_f1, 0.3)
 
 # ifrecallcan
 if recall < 0.3 and precision > 0.5:
 bonus_f1 = max(bonus_f1, 0.3)
 
 # ifrecognitionfeature
 if precision < 0.3 and recall < 0.3 and tp > 0:
 bonus_f1 = max(bonus_f1, 0.1)

 # computetotal
 if not has_reasoning:
 # reasoningcomputemain_rewardbonus_f1already0
 norm_total = self.w_main * norm_main
 else:
 # scoringmain_reward + bonus_f1
 norm_total = self.w_main * norm_main + self.w_bonus * bonus_f1

 # why_zero 
 if not ro_raw:
 why_zero = "oracle(ro)emptysearch_admet(orig_smiles)failreturnsempty"
 elif not rp_raw:
 why_zero = "pred(rp)emptysearch_admet(opt_smiles)failreturnsempty"
 elif len(eval_feats) == 0:
 why_zero = "evaluationfeaturecheck/"
 else:
 why_zero = ""

 # debug
 detail = {
 "main_reward": main_reward,
 "norm_main": norm_main,
 "bonus_hits": bonus_hits,
 "norm_bonus": norm_bonus,
 "bonus_f1": bonus_f1,
 "bad_feats": sorted(list(bad_feats)),
 "num_eval_feats": len(eval_feats),
 "num_pred_feats": len(pred_feats),
 "eval_feats": sorted(list(eval_feats)),
 "pred_feats": sorted(list(pred_feats)), # predictionfeature
 "feat_dirs": feat_dirs, # feature
 "has_reasoning": has_reasoning, # whetherreasoning
 "reasoning_length": len(reasoning) if reasoning else 0, # reasoninglength
 "why_zero": why_zero,
 # efficiencymetric
 "improved_count": improved_count,
 "degraded_count": degraded_count,
 "improvement_efficiency": improved_count / (improved_count + degraded_count) if (improved_count + degraded_count) > 0 else 0,
 "efficiency_bonus": improvement_efficiency_bonus,
 }
 
 # debugoutput
 print(f"[DEBUG] Sample {idx}: main_reward(raw)={main_reward:.3f}, eval_feats={len(eval_feats)}")
 print(f"[DEBUG] Sample {idx}: improved={improved_count}, degraded={degraded_count}, efficiency={improved_count/(improved_count+degraded_count) if (improved_count+degraded_count)>0 else 0:.2%}")
 print(f"[DEBUG] Sample {idx}: base_score={base_score:.3f}, coverage_bonus={coverage_bonus:.3f}, efficiency_bonus={improvement_efficiency_bonus:.3f}, norm_main={norm_main:.3f}")
 print(f"[DEBUG] Sample {idx}: has_reasoning={has_reasoning}, reasoning_length={len(reasoning) if reasoning else 0}")
 if has_reasoning:
 print(f"[DEBUG] Sample {idx}: bad_feats={sorted(list(bad_feats))}, pred_feats={sorted(list(pred_feats))}, bonus_f1={bonus_f1:.3f}")
 print(f"[DEBUG] Sample {idx}: feat_dirs={feat_dirs}")
 else:
 print(f"[DEBUG] Sample {idx}: No reasoning provided, bonus_f1=0.0")
 
 return norm_total, detail

 def compute_sample(self, orig_smiles: str, opt_smiles: str, reasoning: str, idx: Optional[int] = None) -> Tuple[float, Dict[str, Any]]:
 return self.compute_sample_enhanced(orig_smiles, opt_smiles, reasoning, idx)

############################################################
# 8) interface
############################################################

def compute_reward_with_bad_feats(orig_smiles, opt_smiles, reasoning):
 engine = RewardEngine(w_main=1.0, w_bonus=1.0)
 norm_total, detail = engine.compute_sample(orig_smiles, opt_smiles, reasoning)
 return norm_total, detail

############################################################
# 9) I/O 
############################################################

def extract_orig_smiles(rec_o: Dict[str, Any]) -> str:
 instr = rec_o.get("input", "")
 if not instr:
 return ""
 first = instr.splitlines()[0]
 if ":" in first:
 return first.split(":", 1)[1].strip()
 return first.strip()

def parse_reasoning_and_smiles_from_output(output_text: str) -> Tuple[str, str]:
 if not output_text:
 return "", ""
 txt = output_text.strip()
 txt = re.sub(r"\s*<\s*end\s*>\s*$", "", txt, flags=re.IGNORECASE)

 reasoning = ""
 smiles = ""

 for line in txt.splitlines():
 line = line.strip()
 m_r = re.match(r"^reasoning\s*:\s*(.+)$", line, flags=re.IGNORECASE)
 if m_r and not reasoning:
 reasoning = m_r.group(1).strip()
 continue
 m_s = re.match(r"^optimized[_\s-]*smiles\s*:\s*(.+)$", line, flags=re.IGNORECASE)
 if m_s and not smiles:
 smiles = re.sub(r"\s*<\s*end\s*>\s*$", "", m_s.group(1).strip(), flags=re.IGNORECASE).strip()

 if not reasoning:
 # reasoning"Optimized SMILES"fileend
 m = re.search(r"reasoning\s*:\s*(.+?)(?=optimized[_\s-]*smiles|$)", txt, flags=re.IGNORECASE | re.DOTALL)
 if m:
 reasoning = m.group(1).strip()
 if not smiles:
 m = re.search(r"optimized[_\s-]*smiles\s*:\s*([^\s]+)", txt, flags=re.IGNORECASE)
 if m:
 smiles = m.group(1).strip()

 # processmoleculeSMILES
 if smiles and '.' in smiles:
 smiles = handle_multiple_molecules(smiles)

 return reasoning, smiles

def handle_multiple_molecules(smiles: str) -> str:
 """
 processcontainsmoleculeSMILESstring
 
 Args:
 smiles: containsmoleculeSMILESstring
 
 Returns:
 processmoleculeSMILES
 """
 if not smiles or '.' not in smiles:
 return smiles
 
 # splitmolecule
 molecules = smiles.split('.')
 
 # filteremptystring
 molecules = [mol.strip() for mol in molecules if mol.strip()]
 
 if not molecules:
 return smiles
 
 # choosestrategy
 # 1. choosemoleculetargetmolecule
 # 2. iflengthsamechoosefirst
 longest_molecule = max(molecules, key=len)
 
 print(f"[SMILES-PROCESSING] detection{len(molecules)}moleculechoose: {longest_molecule[:50]}...")
 
 return longest_molecule

def handle_multiple_molecules_advanced(smiles: str, strategy: str = "longest") -> str:
 """
 moleculeprocessstrategy
 
 Args:
 smiles: containsmoleculeSMILESstring
 strategy: choosestrategy ("longest", "first", "most_complex", "drug_like")
 
 Returns:
 processmoleculeSMILES
 """
 if not smiles or '.' not in smiles:
 return smiles
 
 # splitmolecule
 molecules = smiles.split('.')
 molecules = [mol.strip() for mol in molecules if mol.strip()]
 
 if not molecules:
 return smiles
 
 if len(molecules) == 1:
 return molecules[0]
 
 print(f"[SMILES-PROCESSING] detection{len(molecules)}moleculeusestrategy: {strategy}")
 
 try:
 from rdkit import Chem
 from rdkit.Chem import Descriptors
 
 # computeeachmoleculeproperty
 molecule_scores = []
 for i, mol_smiles in enumerate(molecules):
 try:
 mol = Chem.MolFromSmiles(mol_smiles)
 if mol is not None:
 # computemoleculeproperty
 mw = Descriptors.MolWt(mol)
 logp = Descriptors.MolLogP(mol)
 tpsa = Descriptors.TPSA(mol)
 num_atoms = mol.GetNumAtoms()
 num_rings = Descriptors.RingCount(mol)
 
 # compute
 if strategy == "drug_like":
 # drugsimilarscoringbased onLipinskirule
 score = 0
 if 150 <= mw <= 500: score += 1
 if logp <= 5: score += 1
 if tpsa <= 140: score += 1
 if Descriptors.NumHDonors(mol) <= 5: score += 1
 if Descriptors.NumHAcceptors(mol) <= 10: score += 1
 elif strategy == "most_complex":
 # scoring
 score = num_atoms + num_rings * 2 + len(mol_smiles)
 elif strategy == "longest":
 # lengthscoring
 score = len(mol_smiles)
 else: # "first"
 score = i
 
 molecule_scores.append((score, mol_smiles, mw, logp, tpsa))
 print(f" molecule{i+1}: MW={mw:.1f}, LogP={logp:.2f}, TPSA={tpsa:.1f}, ={score}")
 else:
 print(f" molecule{i+1}: invalidSMILES")
 except Exception as e:
 print(f" molecule{i+1}: parseerror - {e}")
 
 if molecule_scores:
 # choosemolecule
 best_score, best_smiles, mw, logp, tpsa = max(molecule_scores, key=lambda x: x[0])
 print(f" choosemolecule: MW={mw:.1f}, LogP={logp:.2f}, TPSA={tpsa:.1f}, ={best_score}")
 return best_smiles
 else:
 # ifRDKitparsefailstrategy
 return max(molecules, key=len)
 
 except ImportError:
 print(" RDKitinstallusestrategy")
 return max(molecules, key=len)

def extract_pred_smiles(rec_p: Dict[str, Any]) -> str:
 for key, val in rec_p.items():
 if "smiles" in key.lower() and isinstance(val, str):
 s = val.strip()
 if s:
 return s
 out = rec_p.get("output", "")
 if isinstance(out, str) and out.strip():
 _r, s = parse_reasoning_and_smiles_from_output(out)
 return s or ""
 return ""

def extract_optimized_smiles(rec: Dict[str, Any]) -> str:
 """JSONLrecordextractoptimizationSMILES"""
 # outputextract
 out = rec.get("output", "")
 if isinstance(out, str) and out.strip():
 _r, s = parse_reasoning_and_smiles_from_output(out)
 if s:
 return s
 
 #
 for key, val in rec.items():
 if "smiles" in key.lower() and isinstance(val, str):
 s = val.strip()
 if s:
 return s
 
 return ""

def extract_reasoning(rec: Dict[str, Any]) -> str:
 """JSONLrecordextractreasoning"""
 # outputextract
 out = rec.get("output", "")
 if isinstance(out, str) and out.strip():
 r, _s = parse_reasoning_and_smiles_from_output(out)
 if r:
 return r
 
 #
 for key, val in rec.items():
 if "reasoning" in key.lower() and isinstance(val, str):
 s = val.strip()
 if s:
 return s
 
 return ""

def _extract_from_output_text(prec: Dict[str, Any]) -> Tuple[str, str]:
 """
 prec['output'] parse reasoning optimization SMILES
 rule
 - reasoning: 'reasoning:' 'optimized_smiles' 
 - SMILES: 'Optimized_SMILES:' 'Optimized SMILES:' "firstempty token"
 <END>/<end> and
 """
 raw = prec.get("output", "")
 if not isinstance(raw, str):
 return "", ""

 txt = raw.strip()

 #
 txt = re.sub(r"\s*<\s*end\s*>\s*$", "", txt, flags=re.IGNORECASE)

 # 1) reasoning optimized_smiles/
 m_r = re.search(
 r"(?is)\breasoning\s*:\s*(.+?)(?:\n+\s*(?:optimized[_\s-]*smiles|smiles)\s*:|$)",
 txt
 )
 reasoning = m_r.group(1).strip() if m_r else ""

 # 2) SMILES: first token
 opt_smiles = ""
 for pat in (r"(?im)^\s*optimized[_\s-]*smiles\s*:\s*(.+)$",
 r"(?im)^\s*smiles\s*:\s*(.+)$"):
 m = re.search(pat, txt)
 if m:
 line_rest = m.group(1).strip()
 # <END> comment/
 line_rest = re.sub(r"\s*<\s*end\s*>.*$", "", line_rest, flags=re.IGNORECASE).strip()
 # firstempty
 first_token = line_rest.split()[0]
 # ""
 first_token = first_token.strip(".,;")
 opt_smiles = first_token
 break

 # processmoleculeSMILES
 if opt_smiles and '.' in opt_smiles:
 opt_smiles = handle_multiple_molecules(opt_smiles)

 return reasoning, opt_smiles


############################################################
# 10) RewardEngine
############################################################

def read_reasoning_file(reasoning_path: str) -> Dict[int, str]:
 """
 readinferencefilesupportsformat
 1. "=== Line X ===" formatinference
 2. reasoningformat
 """
 reasoning_dict = {}
 try:
 with open(reasoning_path, "r", encoding="utf-8") as fr:
 content = fr.read()
 
 # checkwhethercontains "=== Line X ===" format
 if "=== Line" in content:
 # format1 "=== Line X ===" split
 sections = re.split(r'=== Line (\d+) ===', content)
 
 for i in range(1, len(sections), 2):
 if i + 1 < len(sections):
 line_num = int(sections[i])
 reasoning_text = sections[i + 1].strip()
 if reasoning_text:
 reasoning_dict[line_num] = reasoning_text
 else:
 # format2reasoning
 lines = content.strip().split('\n')
 for line_num, line in enumerate(lines, start=1):
 reasoning_text = line.strip()
 if reasoning_text: # skipempty
 reasoning_dict[line_num] = reasoning_text
 
 except Exception as e:
 print(f"[WARN] Failed to read reasoning file: {e}")
 
 return reasoning_dict

def main_from_extracted_enhanced(
 orig_jsonl: str,
 after_smi_path: str,
 out_path: str,
 reasoning_path: str = None,
 w_main: float = 1.0,
 w_bonus: float = 1.0,
 golden_path: str = None,
 calibrate_with_golden: bool = False
):
 """
 JSONLfileextractmolecule
 
 description
 reasoning scoringbased onfeaturerecognitionpredictionF1
 """
 engine = RewardEngine(
 w_main=w_main, 
 w_bonus=w_bonus
 )
 
 total_norm_main = 0.0
 total_norm_bonus = 0.0
 total_f1 = 0.0
 total_norm_total = 0.0
 count = 0

 skipped_invalid_after = 0
 paired = 0

 # statistics
 bad_feat_stats = {}
 pred_feat_stats = {}
 reasoning_length_stats = []
 f1_score_distribution = []
 feature_coverage_stats = {}
 has_reasoning_count = 0 # statisticsreasoningsample

 # readoptimizationSMILESfile
 with open(after_smi_path, "r", encoding="utf-8") as f_smiles:
 opt_smiles_list = [line.strip() for line in f_smiles if line.strip()]
 
 # readreasoningfile
 reasoning_list = []
 if reasoning_path and os.path.exists(reasoning_path):
 with open(reasoning_path, "r", encoding="utf-8") as f_reasoning:
 reasoning_list = [line.strip() for line in f_reasoning if line.strip()]
 
 with open(orig_jsonl, "r", encoding="utf-8") as fo, \
 open(out_path, "w", encoding="utf-8") as fw:

 for idx, lo in enumerate(fo):
 line_num = idx + 1 # 1start
 paired += 1
 
 # original JSON
 try:
 orec = json.loads(lo)
 except Exception:
 print(f"[WARN] sample {idx}: bad JSON in orig file, skip")
 continue

 # original SMILES
 orig_smiles = extract_orig_smiles(orec)

 # externalfilereadoptimizationSMILES
 if idx < len(opt_smiles_list):
 opt_smiles = opt_smiles_list[idx]
 else:
 print(f"[WARN] sample {idx}: no corresponding SMILES in after file, skip")
 continue

 # after whether SMILESskip
 if not _is_valid_smiles(opt_smiles):
 skipped_invalid_after += 1
 print(f"[SKIP] sample {idx}: invalid after SMILES -> {opt_smiles[:80]}")
 continue

 # externalfilereadreasoning
 if idx < len(reasoning_list):
 reasoning = reasoning_list[idx]
 else:
 reasoning = ""

 # scoring
 norm_total, detail = engine.compute_sample(orig_smiles, opt_smiles, reasoning, idx=idx)

 # statistics
 reasoning_length_stats.append(len(reasoning))
 f1_score_distribution.append(detail["bonus_f1"])
 
 # statisticswhetherreasoning
 if detail["has_reasoning"]:
 has_reasoning_count += 1
 
 # statisticsfeature
 for feat in detail["bad_feats"]:
 bad_feat_stats[feat] = bad_feat_stats.get(feat, 0) + 1
 
 # statisticspredictionfeature
 for feat in detail["pred_feats"]:
 pred_feat_stats[feat] = pred_feat_stats.get(feat, 0) + 1
 
 # statisticsfeatureusefeature
 core_eval_feats = [f for f in detail["eval_feats"] if f in CORE_FEATURES]
 coverage = len(core_eval_feats) / len(CORE_FEATURES)
 feature_coverage_stats[idx] = coverage

 # print
 print(f"\n=== Sample {idx} ===")
 print("orig_smiles:", orig_smiles)
 print("opt_smiles :", opt_smiles)
 print("reasoning :", reasoning[:200] + "..." if len(reasoning) > 200 else reasoning)
 print("detail :", detail)

 # total
 total_norm_main += detail["norm_main"]
 total_norm_bonus += detail["norm_bonus"]
 total_f1 += detail["bonus_f1"]
 total_norm_total += norm_total
 count += 1

 # outputaveragestatistics
 if count > 0:
 averages = {
 "avg_main_reward": round(total_norm_main / count, 4),
 "avg_bonus": round(total_norm_bonus / count, 4),
 "avg_bonus_f1": round(total_f1 / count, 4),
 "avg_total_reward": round(total_norm_total / count, 4),
 "w_main": w_main,
 "w_bonus": w_bonus,
 "paired": paired,
 "used_pairs": count,
 "skipped_invalid_after": skipped_invalid_after,
 "reasoning_loaded": 0, # JSONLextractno need toexternalreasoningfile
 # statistics
 "avg_reasoning_length": round(sum(reasoning_length_stats) / len(reasoning_length_stats), 2),
 "f1_score_stats": {
 "min": round(min(f1_score_distribution), 4),
 "max": round(max(f1_score_distribution), 4),
 "median": round(sorted(f1_score_distribution)[len(f1_score_distribution)//2], 4),
 "std": round((sum((x - sum(f1_score_distribution)/len(f1_score_distribution))**2 for x in f1_score_distribution) / len(f1_score_distribution))**0.5, 4)
 },
 "top_bad_features": dict(sorted(bad_feat_stats.items(), key=lambda x: x[1], reverse=True)[:10]),
 "top_pred_features": dict(sorted(pred_feat_stats.items(), key=lambda x: x[1], reverse=True)[:10]),
 "avg_feature_coverage": round(sum(feature_coverage_stats.values()) / len(feature_coverage_stats), 4),
 "has_reasoning_count": has_reasoning_count,
 "has_reasoning_ratio": round(has_reasoning_count / count, 4) if count > 0 else 0.0
 }

 # ===== Golden alignmentevaluationoptional=====
 def _rankdata(vals: List[float]) -> List[float]:
 # averageprocess ties
 pairs = sorted((v, i) for i, v in enumerate(vals))
 ranks = [0.0] * len(vals)
 i = 0
 while i < len(pairs):
 j = i
 v = pairs[i][0]
 while j < len(pairs) and pairs[j][0] == v:
 j += 1
 # average1start
 avg_rank = (i + 1 + j) / 2.0
 for k in range(i, j):
 ranks[pairs[k][1]] = avg_rank
 i = j
 return ranks

 def _pearson(x: List[float], y: List[float]) -> float:
 n = min(len(x), len(y))
 if n == 0:
 return 0.0
 x = x[:n]; y = y[:n]
 mx = sum(x)/n; my = sum(y)/n
 num = sum((a-mx)*(b-my) for a,b in zip(x,y))
 denx = sum((a-mx)**2 for a in x)
 deny = sum((b-my)**2 for b in y)
 if denx <= 1e-12 or deny <= 1e-12:
 return 0.0
 return num / (denx**0.5 * deny**0.5)

 def _spearmanr(xs: List[float], ys: List[float]) -> float:
 rx = _rankdata(xs)
 ry = _rankdata(ys)
 return _pearson(rx, ry)

 def _tier(v: float) -> int:
 if v <= 0.33: return 0
 if v <= 0.66: return 1
 return 2

 def _tier_agree(pred: List[float], gold: List[float]) -> float:
 n = min(len(pred), len(gold))
 if n == 0:
 return 0.0
 hit = 0
 for i in range(n):
 if _tier(pred[i]) == _tier(gold[i]):
 hit += 1
 return round(hit / n, 4)

 # Golden alignmentremovedependency lms_judge
 fw.write(json.dumps(averages, ensure_ascii=False) + "\n")

 print(f"✅ Averages saved to {out_path}")
 print(f" paired={paired}, used={count}, skipped_invalid_after={skipped_invalid_after}")
 print(f" reasoning_loaded=0") # JSONLextractno need toexternalreasoningfile
 print(f" avg_bonus_f1={averages['avg_bonus_f1']}, f1_stats={averages['f1_score_stats']}")

# replacemain_from_extractedfunction
def main_from_extracted(orig_jsonl: str, after_smi_path: str, out_path: str, reasoning_path: str = None, 
 w_main: float = 1.0, w_bonus: float = 1.0,
 golden_path: str = None,
 calibrate_with_golden: bool = False):
 return main_from_extracted_enhanced(orig_jsonl, after_smi_path, out_path, reasoning_path, 
 w_main, w_bonus,
 golden_path, calibrate_with_golden)

############################################################
# 11) 
############################################################

if __name__ == "__main__":
 main_from_extracted(
 #orig_jsonl="${DATA_DIR:-/path/to/data}/val/sample_300_deduplicated_correct.jsonl", 
 #orig_jsonl="${DATA_DIR:-/path/to/data}/ood_drug_admet_converted.jsonl",
 orig_jsonl="${DATA_DIR:-/path/to/data}/val/converted_reasoning_test5_fixed.jsonl",
 # originalmolecule
 #after_smi_path="${DATA_DIR:-/path/to/data}/result/pretrain_to_sft.txt", 
 #after_smi_path="${DATA_DIR:-/path/to/data}/result/pretrain_to_sft_molnet_smiles.txt", # extractoptimizationSMILES
 #after_smi_path="${DATA_DIR:-/path/to/data}/result/base.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/result/base_clean_smiles.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/result/ether0_clean_smiles.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/result/ether0_ood_smiles.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/result/rl1000_smiles.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/result/rl1200_smiles.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/result/pretrain_to_sft_molnet_clean2_smiles.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/result/rl100_smiles.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/result/pretrain_to_sft_molnet_clean_formatted_smiles.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/rl/rl200_smiles.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/change/pretrain_to_sft4_smiles.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/change/rl1600_smiles.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/change/base_smiles.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/ultrachat-clean/separated_output/smiles.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/change/ether0_smiles.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/rl/pretrain_to_sft3_smiles.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/change/sft_smiles.txt",
 after_smi_path="${DATA_DIR:-/path/to/data}/rl/rl600_smiles.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/change/noreasoning_smiles.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/change/sft_pretrain_llm8b2000_smiles.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/val/output/generated_results_smiles.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/change/sft_and_pretrain_smiles3.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/change/rl1400_smiles.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/val/output/generated_results_smiles.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/ultrachat-clean/separated_output_base/smiles.txt",
 #after_smi_path="${DATA_DIR:-/path/to/data}/result/pretrain_to_sft_ood_smiles.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/result/pretrain_to_sft_reasoning.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/result/base_reasoning.txt", # inference
 #reasoning_path="${DATA_DIR:-/path/to/data}/result/base_clean_reasoning.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/result/ether0_clean_reasoning.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/result/ether0_ood_reasoning.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/result/rl1000_reasoning.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/result/rl100_reasoning.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/result/pretrain_to_sft_molnet_clean_formatted_reasoning.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/change/sft_pretrain_llm8b2000_reasoning.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/change/sft_and_pretrain_reasoning3.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/change/rl1400_reasoning.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/rl/rl200_reasoning.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/change/pretrain_to_sft4_reasoning.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/change/rl1600_reasoning.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/change/base_reasoning.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/ultrachat-clean/separated_output/reasoning.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/change/ether0_reasoning.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/change/admet/sft_pretrain_reasoning.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/change/sft_reasoning.txt",
 reasoning_path="${DATA_DIR:-/path/to/data}/rl/rl600_reasoning.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/rl/pretrain_to_sft3_reasoning.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/val/output/generated_results_rationale.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/val/output/generated_results_rationale.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/ultrachat-clean/separated_output_base/reasoning.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/result/rl1200_reasoning.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/result/pretrain_to_sft_molnet_clean2_reasoning.txt",
 #reasoning_path="${DATA_DIR:-/path/to/data}/result/pretrain_to_sft_ood_reasoning.txt",
 #out_path="${DATA_DIR:-/path/to/data}/admet/bpretrain_to_sft_results.jsonl",
 #out_path="${DATA_DIR:-/path/to/data}/admet/rl1200.jsonl",
 #out_path="${DATA_DIR:-/path/to/data}/admet/rl100.jsonl",
 #out_path="${DATA_DIR:-/path/to/data}/admet/pretrain_to_sft_molnet_clean_formatted.jsonl",
 #out_path="${DATA_DIR:-/path/to/data}/admet/deepseek_results.jsonl",
 #out_path="${DATA_DIR:-/path/to/data}/change/admet/pretrain_to_sft3.jsonl",
 #out_path="${DATA_DIR:-/path/to/data}/change/admet/sft.jsonl",
 out_path="${DATA_DIR:-/path/to/data}/admet/rl600.jsonl",
 #out_path="${DATA_DIR:-/path/to/data}/change/admet/pretrain_to_sft10.jsonl",
 #out_path="${DATA_DIR:-/path/to/data}/change/admet/sft_pretrain_llm8b2000.jsonl",
 #out_path="${DATA_DIR:-/path/to/data}/change/admet/sft_and_pretrain3.jsonl",
 #out_path="${DATA_DIR:-/path/to/data}/change/admet/rl1400.jsonl",
 #out_path="${DATA_DIR:-/path/to/data}/admet/base_test.jsonl",
 w_main=1.0,
 w_bonus=1.0
 )

