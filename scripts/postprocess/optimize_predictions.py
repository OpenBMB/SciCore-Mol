#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimizationpredictionresultscript - JSONL file pred optimization

 python scripts/optimize_predictions.py \
 --prediction_dir ${SCICORE_ROOT:-/path/to/scicore-mol}/1228results_baseline/LlaSMol-Mistral-7B-merged
"""

import argparse
import json
import re
import sys
from pathlib import Path

# directorypath
_project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_project_root))

# tasktypedefine eval_smolinstruct.py consistent
SMILES_TASKS = {
 "forward_synthesis",
 "retrosynthesis",
 "molecule_generation",
 "name_conversion-i2s",
}
TEXT_TASKS = {"molecule_captioning"}
FORMULA_ELEMENT_TASKS = {"name_conversion-i2f", "name_conversion-s2f"}
FORMULA_SPLIT_TASKS = {"name_conversion-s2i"}
NUMBER_TASKS = {"property_prediction-esol", "property_prediction-lipo"}
BOOLEAN_TASKS = {
 "property_prediction-bbbp",
 "property_prediction-clintox",
 "property_prediction-hiv",
 "property_prediction-sider",
}

# regex
SMILES_TOKEN_RE = re.compile(r"([A-Za-z0-9@+\-\[\]\(\)=#\\/%.]+)")
NUMBER_TOKEN_RE = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")
# matchcontainsvalueSMILES
# matchformat-4.07, -4.883, 4.5, -20.86 
NUMBER_WITH_DECIMAL_OR_NEGATIVE_RE = re.compile(r"[-+]?\d+\.\d+|[-+]\d+\.\d*[eE][-+]?\d+")


def _is_task_file(p: Path) -> bool:
 """judgewhethertaskfile"""
 return p.is_file() and p.suffix == ".jsonl" and p.name != "eval_summary.json"


def _extract_based_on_gold_format(text: str, gold: str, task_name: str) -> str:
 """
 according togoldformattextextractsimilar
 
 Args:
 text: extractanswer_onlyraw_output
 gold: goldforjudgeformat
 task_name: task
 
 Returns:
 extractpredictionresult
 """
 if not text or not gold:
 return ""
 
 text = str(text).strip()
 gold = str(gold).strip()
 
 # 1. valuetaskgoldextractextractcontains
 if task_name in NUMBER_TASKS:
 # removeSMILESlabelextractSMILES
 text_without_smiles = re.sub(r"<SMILES[^>]*>.*?</SMILES>", "", text, flags=re.IGNORECASE | re.DOTALL)
 
 # SMILEScontains
 matches = NUMBER_WITH_DECIMAL_OR_NEGATIVE_RE.findall(text_without_smiles)
 if matches:
 return matches[-1] # returnslast
 
 # ifSMILES
 matches = NUMBER_WITH_DECIMAL_OR_NEGATIVE_RE.findall(text)
 if matches:
 return matches[-1]
 
 # ifextractallchoosevaluecontains
 all_numbers = NUMBER_TOKEN_RE.findall(text_without_smiles)
 likely_numbers = [n for n in all_numbers if '.' in n or n.startswith('-') or n.startswith('+')]
 if likely_numbers:
 return likely_numbers[-1]
 
 #
 all_numbers = NUMBER_TOKEN_RE.findall(text)
 likely_numbers = [n for n in all_numbers if '.' in n or n.startswith('-') or n.startswith('+')]
 if likely_numbers:
 return likely_numbers[-1]
 
 return ""
 
 # 2. SMILEStaskgoldSMILESstringextractSMILES
 if task_name in SMILES_TASKS:
 # extract<SMILES>labelsupportslabel
 smiles_pattern = re.compile(r"<SMILES[^>]*>(.*?)</SMILE[S]?>", re.IGNORECASE | re.DOTALL)
 matches = smiles_pattern.findall(text)
 if matches:
 # returnslastmatch
 content = matches[-1].strip()
 if content:
 return content
 
 # iflabelextractSMILESstring
 m = SMILES_TOKEN_RE.search(text)
 if m:
 return m.group(1)
 return ""
 
 # 3. taskgoldYes/NoextractYes/No
 if task_name in BOOLEAN_TASKS:
 # extract<BOOLEAN>label
 boolean_pattern = re.compile(r"<BOOLEAN[^>]*>(.*?)</BOOLEAN[T]?>", re.IGNORECASE | re.DOTALL)
 matches = boolean_pattern.findall(text)
 if matches:
 content = matches[-1].strip()
 if content:
 # Yes/No
 content_lower = content.lower()
 if "yes" in content_lower or "true" in content_lower or "toxic" in content_lower:
 return "Yes"
 elif "no" in content_lower or "false" in content_lower or "non-toxic" in content_lower:
 return "No"
 
 # extractYes/Nosize
 yes_no_pattern = re.compile(r"\b(Yes|No)\b", re.IGNORECASE)
 matches = yes_no_pattern.findall(text)
 if matches:
 # returnslastmatch
 result = matches[-1]
 # Yes/No
 return "Yes" if result.lower() == "yes" else "No"
 return ""
 
 # 4. taskgoldextractSMILESlabel
 if task_name in TEXT_TASKS:
 # removeSMILESlabel
 text_cleaned = re.sub(r"<SMILES[^>]*>.*?</SMILES>", "", text, flags=re.IGNORECASE | re.DOTALL)
 text_cleaned = text_cleaned.strip()
 if text_cleaned:
 return text_cleaned
 # ifreturnslabel
 return re.sub(r"<[^>]+>", "", text).strip()
 
 # 5. task
 if task_name in FORMULA_ELEMENT_TASKS:
 # name_conversion-i2f, name_conversion-s2f: <MOLFORMULA>labelextractmolecule
 # extract<MOLFORMULA>labelsupportslabelincludingempty
 molformula_pattern = re.compile(r"<MOLFORMULA[^>]*>(.*?)</MOLFORMATULA[T]?\s*>", re.IGNORECASE | re.DOTALL)
 matches = molformula_pattern.findall(text)
 if matches:
 # returnslastmatch
 content = matches[-1].strip()
 if content:
 return content
 
 # ifMOLFORMULAlabelreturnsemptystringoriginalpred
 return ""
 
 if task_name in FORMULA_SPLIT_TASKS:
 # name_conversion-s2i: <IUPAC>labelextractIUPAC
 # extract<IUPAC>labelsupportslabel
 iupac_pattern = re.compile(r"<IUPAC[^>]*>(.*?)</IUPAC[C]?\s*>", re.IGNORECASE | re.DOTALL)
 matches = iupac_pattern.findall(text)
 if matches:
 # returnslastmatch
 content = matches[-1].strip()
 if content:
 return content
 
 # ifIUPAClabelreturnsemptystringoriginalpred
 return ""
 
 # defaultreturns
 return text.strip()


def optimize_predictions_in_file(file_path: Path, backup: bool = True) -> dict:
 """
 optimizationfileallprediction
 
 Args:
 file_path: JSONL filepath
 backup: whethercreatebackupfile
 
 Returns:
 statisticsdict
 """
 stats = {
 "total": 0,
 "optimized": 0,
 "failed": 0,
 "unchanged": 0,
 }
 
 # createbackup
 if backup:
 backup_path = file_path.with_suffix(file_path.suffix + ".backup")
 if backup_path.exists():
 print(f"[WARN] backupfileskipbackup: {backup_path}")
 else:
 import shutil
 shutil.copy2(file_path, backup_path)
 print(f"[INFO] createbackup: {backup_path}")
 
 # readallrecord
 records = []
 with open(file_path, "r", encoding="utf-8") as f:
 for line in f:
 line = line.strip()
 if not line:
 continue
 try:
 records.append(json.loads(line))
 stats["total"] += 1
 except json.JSONDecodeError as e:
 print(f"[ERROR] parse JSON fail ( {stats['total']+1}): {e}")
 stats["failed"] += 1
 
 # defaulttaskfile
 default_task_name = file_path.stem
 
 print(f"[INFO] processfile: {file_path.name}, defaulttask: {default_task_name}, record: {stats['total']}")
 
 # optimizationrecord pred 
 for i, record in enumerate(records):
 original_pred = record.get("pred", None)
 raw_output = record.get("raw_output", "")
 answer_only = record.get("answer_only", None)
 
 # taskuserecord task 
 task_name = record.get("task", default_task_name)
 gold = record.get("gold", "")
 
 # extractpredictionaccording togoldformat
 try:
 # use answer_onlyuse raw_output
 source_text = None
 if answer_only and isinstance(answer_only, str) and answer_only.strip():
 source_text = answer_only
 elif raw_output and isinstance(raw_output, str) and raw_output.strip():
 source_text = raw_output
 
 if source_text:
 optimized_pred = _extract_based_on_gold_format(source_text, gold, task_name)
 else:
 optimized_pred = original_pred
 
 # ifextractsuccessupdate pred 
 if optimized_pred and optimized_pred.strip():
 if optimized_pred != original_pred:
 record["pred"] = optimized_pred
 stats["optimized"] += 1
 else:
 stats["unchanged"] += 1
 else:
 # ifextractfailoriginal predif
 if not original_pred or not str(original_pred).strip():
 stats["failed"] += 1
 else:
 stats["unchanged"] += 1
 except Exception as e:
 print(f"[ERROR] optimizationfail (record {i+1}): {e}")
 stats["failed"] += 1
 
 # file
 with open(file_path, "w", encoding="utf-8") as f:
 for record in records:
 f.write(json.dumps(record, ensure_ascii=False) + "\n")
 
 return stats


def main():
 parser = argparse.ArgumentParser(description="optimizationpredictionresult")
 parser.add_argument(
 "--prediction_dir",
 type=str,
 required=True,
 help="containspredictionresult jsonl filedirectory",
 )
 parser.add_argument(
 "--no_backup",
 action="store_true",
 help="createbackupfile",
 )
 parser.add_argument(
 "--file_pattern",
 type=str,
 default="",
 help="processmatchmodefileoptional '*.jsonl'",
 )
 
 args = parser.parse_args()
 
 pred_dir = Path(args.prediction_dir).expanduser().resolve()
 if not pred_dir.exists():
 print(f"[ERROR] directory: {pred_dir}")
 sys.exit(1)
 
 if not pred_dir.is_dir():
 print(f"[ERROR] directory: {pred_dir}")
 sys.exit(1)
 
 # alltaskfile
 files = [p for p in pred_dir.iterdir() if _is_task_file(p)]
 
 if args.file_pattern:
 from fnmatch import fnmatch
 files = [f for f in files if fnmatch(f.name, args.file_pattern)]
 
 if not files:
 print(f"[WARN] directorytaskfile{pred_dir}")
 sys.exit(0)
 
 print(f"[INFO] {len(files)} fileneedsprocess")
 
 # processeachfile
 total_stats = {
 "total": 0,
 "optimized": 0,
 "failed": 0,
 "unchanged": 0,
 }
 
 for file_path in sorted(files):
 print(f"\n{'='*60}")
 stats = optimize_predictions_in_file(file_path, backup=not args.no_backup)
 
 # totalstatistics
 for key in total_stats:
 total_stats[key] += stats[key]
 
 print(f"[INFO] complete: {file_path.name}")
 print(f" - total: {stats['total']}")
 print(f" - optimization: {stats['optimized']}")
 print(f" - : {stats['unchanged']}")
 print(f" - fail: {stats['failed']}")
 
 # printtotalstatistics
 print(f"\n{'='*60}")
 print(f"[INFO] complete")
 print(f" - processfile: {len(files)}")
 print(f" - totalrecord: {total_stats['total']}")
 print(f" - optimization: {total_stats['optimized']}")
 print(f" - : {total_stats['unchanged']}")
 print(f" - fail: {total_stats['failed']}")


if __name__ == "__main__":
 main()

