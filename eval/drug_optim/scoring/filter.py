import json
import os

# inputoutputpath
input_path = "${DATA_DIR:-/path/to/data}/reasoning_process/reasoning_process.jsonl"
output_path = "${DATA_DIR:-/path/to/data}/reasoning_process/reasoning_process_cleaned.jsonl"

# needsADMETkey
KEY_FEATURES = [
 #
 "Caco-2 Permeability", "MDCK Permeability", "HIA", "F20%", "F30%", "F50%",
 #
 "BBB", "Pgp inhibitor", "Pgp substrate", "BCRP inhibitor", "VDss", "PPB",
 #
 "CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4", "HLM Stability",
 #
 "CLplasma", "T1/2",
 #
 "hERG Blockers", "DILI", "Human Hepatotoxicity", "AMES Toxicity", "Genotoxicity",
 "Rat Oral Acute Toxicity", "Drug-induced Nephrotoxicity",
 "Drug-induced Neurotoxicity", "Ototoxicity", "Hematotoxicity", "Skin Sensitization",
 # class
 "Lipinski Rule", "Pfizer Rule", "GSK Rule", "GoldenTriangle", "QED", "Fsp3",
 #
 "SAscore", "MCE-18"
]

###############################################################################
# evaluationthresholdfor bad feature judgethreshold
# - thresholdconfigevaluationrun
# - /threshold
###############################################################################

# valuemapping
_QUAL_TO_SCORE = {
 "++++": 4,
 "+++": 3,
 "++": 2,
 "+": 1,
 "+/-": 0.5,
 "-": 0,
 "--": -1,
 "---": -2
}


def _to_float(value):
 """stringvaluevaluesupports +/++/+++ label"""
 if value is None:
 return None
 s = str(value).strip()
 if s in _QUAL_TO_SCORE:
 return float(_QUAL_TO_SCORE[s])
 try:
 return float(s)
 except Exception:
 return None


def check_item(item, value) -> bool:
 """
 judgemetricwhether“threshold”
 item: (flag, threshold, msg)
 """
 flag, threshold, _msg = item
 v = _to_float(value)
 t = _to_float(threshold)
 if flag in (">", ">=", "ge", "gte"):
 return v is not None and t is not None and v >= t
 if flag in ("<", "<=", "le", "lte"):
 return v is not None and t is not None and v <= t
 if flag in ("=", "==", "eq"):
 return str(value).strip() == str(threshold).strip()
 if flag in ("!=", "neq"):
 return str(value).strip() != str(threshold).strip()
 if flag == "contains":
 return str(threshold).strip() in str(value)
 return False


# checks {category: {feature: (flag, threshold, msg)}}
checks = {
 "toxicity": {
 "AMES Toxicity": (">=", 0.5, "AMES "),
 "Genotoxicity": (">=", 0.5, ""),
 "DILI": (">=", 0.5, "DILI "),
 "Human Hepatotoxicity": (">=", 0.5, ""),
 "hERG Blockers": (">=", 0.5, "hERG ")
 },
 "absorption": {
 "Caco-2 Permeability": ("<=", -5.0, "Caco-2 "),
 "MDCK Permeability": ("<=", 0.0, "MDCK "),
 "HIA": ("<=", 0.0, "HIA "),
 "F20%": ("<=", 0.0, "F20% "),
 "F30%": ("<=", 0.0, "F30% "),
 "F50%": ("<=", 0.0, "F50% ")
 },
 "distribution": {
 "Pgp inhibitor": (">=", "++", "P-gp "),
 "Pgp substrate": (">=", "++", "P-gp "),
 "BCRP inhibitor": (">=", "++", "BCRP "),
 "PPB": (">=", 95, "")
 },
 "metabolism": {
 "CYP1A2 inhibitor": (">=", "++", "CYP1A2 "),
 "CYP2C19 inhibitor": (">=", "++", "CYP2C19 "),
 "CYP2C9 inhibitor": (">=", "++", "CYP2C9 "),
 "CYP2D6 inhibitor": (">=", "++", "CYP2D6 "),
 "CYP3A4 inhibitor": (">=", "++", "CYP3A4 ")
 }
}


def clean_admet_profile(admet_text: str) -> str:
 """
 originalADMET Profilekey
 """
 cleaned_lines = []
 for line in admet_text.split("\n"):
 if any(key in line for key in KEY_FEATURES):
 cleaned_lines.append(line)
 return "\n".join(cleaned_lines)


def main():
 # readdata
 with open(input_path, "r", encoding="utf-8") as fin, \
 open(output_path, "w", encoding="utf-8") as fout:
 for line in fin:
 if not line.strip():
 continue
 sample = json.loads(line)

 original_input = sample.get("input", "")
 # inputSMILESADMET Profile
 parts = original_input.split("ADMET Profile:")

 if len(parts) == 2:
 smiles = parts[0].strip()
 admet_profile = clean_admet_profile(parts[1])
 #
 sample["input"] = f"{smiles}\nADMET Profile:\n{admet_profile}"
 else:
 # ifformat
 sample["input"] = original_input

 fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

 print(f"completeoutputfilesave: {output_path}")


if __name__ == "__main__":
 main()
