#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fixpredictionextractscoring

 python scripts/fix_and_rescore.py \
 --prediction_dir ${SCICORE_ROOT:-/path/to/scicore-mol}/1125results_baseline/LlaSMol-Mistral-7B-merged_fewshot \
 --backup
"""

import argparse
import json
import sys
import re
from pathlib import Path
from typing import Optional

# directorypath
_project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_project_root))

from eval.extract_prediction import extract_prediction_from_raw

# tasktypedefineeval_smolinstruct.pyconsistent
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
FORMULA_TOKEN_RE = re.compile(r"([A-Za-z0-9\(\)\.\+\-]+)")
NUMBER_TOKEN_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
BOOL_TOKEN_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
SMILES_TAG_RE = re.compile(r"<SMILES>\s*([A-Za-z0-9@+\-\[\]\(\)=#\\/%.]+)\s*</SMILES>", re.IGNORECASE)


def _canonical_bool(text: str) -> str:
 """ Yes/No"""
 if not isinstance(text, str):
 text = str(text)
 text = text.strip().lower()
 text = text.rstrip('.,;:!?')
 
 yes_values = {"yes", "y", "true", "t", "1", "positive", "toxic", "unsafe", "harmful"}
 no_values = {"no", "n", "false", "f", "0", "negative", "non-toxic", "non toxic", "nontoxic", "safe", "non-harmful"}
 
 if text in yes_values:
 return "Yes"
 elif text in no_values:
 return "No"
 
 m = BOOL_TOKEN_RE.search(text)
 if m:
 v = m.group(1).lower()
 return "Yes" if v == "yes" else "No"
 
 if "toxic" in text and "non" not in text and "not" not in text:
 return "Yes"
 elif "non-toxic" in text or "nontoxic" in text or ("non" in text and "toxic" in text):
 return "No"
 
 return ""


def _extract_core_answer(text: str, task_name: str) -> str:
 """extract"""
 if not text or not isinstance(text, str):
 return ""
 text = str(text).strip()
 
 if task_name in TEXT_TASKS:
 return text
 
 if task_name in SMILES_TASKS:
 # <SMILES>labelextract
 m = SMILES_TAG_RE.search(text)
 if m:
 return m.group(1).strip()
 
 # iflabel</SMILES>extract
 if "</SMILES>" in text:
 before_close = text.split("</SMILES>")[0]
 # SMILESstring
 matches = list(SMILES_TOKEN_RE.finditer(before_close))
 if matches:
 # lastmatchSMILES
 return matches[-1].group(1).strip()
 
 # extractSMILESmatch
 all_matches = []
 for line in text.splitlines():
 line = line.strip()
 if not line:
 continue
 matches = list(SMILES_TOKEN_RE.finditer(line))
 if matches:
 all_matches.extend(matches)
 
 if all_matches:
 # SMILESstring
 longest_match = max(all_matches, key=lambda m: len(m.group(1)))
 return longest_match.group(1).strip()
 
 return text
 
 if task_name in FORMULA_ELEMENT_TASKS or task_name in FORMULA_SPLIT_TASKS:
 for line in text.splitlines():
 line = line.strip()
 if not line:
 continue
 m = FORMULA_TOKEN_RE.search(line)
 if m:
 return m.group(1)
 return text
 
 if task_name in NUMBER_TASKS:
 m = NUMBER_TOKEN_RE.search(text)
 if m:
 return m.group(0)
 return text
 
 if task_name in BOOLEAN_TASKS:
 return _canonical_bool(text)
 
 # defaultreturnsnon-empty
 for line in text.splitlines():
 line = line.strip()
 if line:
 return line
 return text


def extract_prediction_improved(obj: dict, task_name: str) -> Optional[str]:
 """
 predictionextractfunction
 prioritypred > answer_only > raw_output
 """
 # 1. ifpredvalidcheckwhetherneedsextract
 pred = None
 need_re_extract = False
 
 if pred is not None and isinstance(pred, str) and pred.strip():
 pred_stripped = pred.strip()
 # taskifpred"The"needsextract
 if task_name in BOOLEAN_TASKS:
 bool_result = _canonical_bool(pred_stripped)
 if not bool_result or len(pred_stripped) < 2:
 need_re_extract = True
 else:
 return bool_result
 # SMILEStaskifpredSMILESneedsextract
 elif task_name in SMILES_TASKS:
 if len(pred_stripped) < 5 or not SMILES_TOKEN_RE.search(pred_stripped):
 need_re_extract = True
 else:
 return pred_stripped
 else:
 # taskifpredneedsextract
 if len(pred_stripped) < 2:
 need_re_extract = True
 else:
 return pred_stripped
 else:
 # predemptyNoneneedsextract
 need_re_extract = True
 
 if not need_re_extract:
 return None
 
 # 2. answer_onlyextract
 answer_only = obj.get("answer_only", None)
 if answer_only and isinstance(answer_only, str) and answer_only.strip():
 # taskifanswer_onlycontainsSMILESlabelerroroutputshouldignore
 if task_name in BOOLEAN_TASKS and "<SMILES>" in answer_only:
 # skipraw_outputextract
 pass
 else:
 extracted = _extract_core_answer(answer_only.strip(), task_name)
 if extracted and extracted.strip():
 return extracted
 
 # 3. raw_outputextract
 raw_output = obj.get("raw_output", None)
 if raw_output and isinstance(raw_output, str) and raw_output.strip():
 try:
 extracted = extract_prediction_from_raw(
 raw_output, task_name,
 text_tasks=TEXT_TASKS,
 smiles_tasks=SMILES_TASKS,
 formula_element_tasks=FORMULA_ELEMENT_TASKS,
 formula_split_tasks=FORMULA_SPLIT_TASKS,
 number_tasks=NUMBER_TASKS,
 boolean_tasks=BOOLEAN_TASKS,
 answer_only=answer_only, # answer_only
 )
 if extracted and extracted.strip():
 # taskvalidateextractresult
 if task_name in BOOLEAN_TASKS:
 bool_result = _canonical_bool(extracted.strip())
 if bool_result:
 return bool_result
 # ifextractresultvalidvaluereturnsNone
 return None
 return extracted.strip()
 except Exception as e:
 # ifextractfailanswer_onlyextractif
 if answer_only and task_name not in BOOLEAN_TASKS:
 extracted = _extract_core_answer(answer_only.strip(), task_name)
 if extracted and extracted.strip():
 return extracted
 
 return None


def fix_jsonl_file(jsonl_path: Path, backup: bool = False) -> tuple[int, int]:
 """
 fixjsonlfile
 returns: (fixcount, totalcount)
 """
 if not jsonl_path.exists():
 print(f"[WARN] file: {jsonl_path}")
 return 0, 0
 
 # backup
 if backup:
 backup_path = jsonl_path.with_suffix(jsonl_path.suffix + ".backup")
 if not backup_path.exists():
 import shutil
 shutil.copy2(jsonl_path, backup_path)
 print(f"[INFO] backup: {backup_path}")
 
 task_name = jsonl_path.stem
 fixed_count = 0
 total_count = 0
 fixed_entries = []
 
 # readfix
 with jsonl_path.open("r", encoding="utf-8") as f:
 for line in f:
 line = line.strip()
 if not line:
 continue
 
 try:
 obj = json.loads(line)
 total_count += 1
 
 # gettask
 task_from_obj = obj.get("task", task_name)
 
 # extractprediction
 new_pred = extract_prediction_improved(obj, task_from_obj)
 
 # checkwhetherneedsupdate
 old_pred = obj.get("pred", "")
 old_pred_str = old_pred if old_pred is not None else ""
 
 if new_pred is not None and new_pred.strip():
 # extractresult
 new_pred_str = new_pred.strip()
 if old_pred_str != new_pred_str:
 obj["pred"] = new_pred_str
 fixed_count += 1
 elif old_pred_str and old_pred_str.strip():
 # ifextractfailpredpred
 # needscheckpredwhethervalid
 if task_from_obj in BOOLEAN_TASKS:
 # taskvalidatepredwhethervalid
 bool_result = _canonical_bool(old_pred_str.strip())
 if not bool_result:
 # predinvalidempty
 obj["pred"] = ""
 fixed_count += 1
 else:
 # ifemptypredemptystring
 if old_pred_str != "":
 obj["pred"] = ""
 fixed_count += 1
 
 fixed_entries.append(obj)
 except json.JSONDecodeError as e:
 print(f"[ERROR] JSONparsefail (line {total_count+1}): {e}")
 continue
 except Exception as e:
 print(f"[ERROR] processfail (line {total_count+1}): {e}")
 continue
 
 # file
 if fixed_entries:
 with jsonl_path.open("w", encoding="utf-8") as f:
 for obj in fixed_entries:
 f.write(json.dumps(obj, ensure_ascii=False) + "\n")
 
 return fixed_count, total_count


def main():
 parser = argparse.ArgumentParser(description="fixpredictionextractscoring")
 parser.add_argument(
 "--prediction_dir",
 type=str,
 required=True,
 help="containspredictionresult jsonl filedirectory",
 )
 parser.add_argument(
 "--backup",
 action="store_true",
 help="fixbackuporiginalfile",
 )
 parser.add_argument(
 "--rescore",
 action="store_true",
 help="fixscoring",
 )
 parser.add_argument(
 "--save_json",
 type=str,
 default="",
 help="savescoringresult JSON filepathoptional",
 )
 parser.add_argument(
 "--score_workers",
 type=int,
 default=1,
 help="scoringuseprocess",
 )
 
 args = parser.parse_args()
 
 pred_dir = Path(args.prediction_dir).expanduser().resolve()
 if not pred_dir.exists():
 print(f"errordirectory: {pred_dir}")
 sys.exit(1)
 
 if not pred_dir.is_dir():
 print(f"errordirectory: {pred_dir}")
 sys.exit(1)
 
 # alljsonlfile
 jsonl_files = list(pred_dir.glob("*.jsonl"))
 jsonl_files = [f for f in jsonl_files if not f.name.startswith("_")]
 
 if not jsonl_files:
 print(f"[WARN] jsonlfile: {pred_dir}")
 sys.exit(1)
 
 print(f"[INFO] {len(jsonl_files)} jsonlfile")
 
 # fixeachfile
 total_fixed = 0
 total_samples = 0
 for jsonl_file in sorted(jsonl_files):
 print(f"\n[INFO] process: {jsonl_file.name}")
 fixed, total = fix_jsonl_file(jsonl_file, backup=args.backup)
 total_fixed += fixed
 total_samples += total
 print(f" - fix: {fixed}/{total} record")
 
 print(f"\n[INFO] fixcomplete: fix {total_fixed}/{total_samples} record")
 
 # scoring
 if args.rescore:
 print("\n[INFO] startscoring...")
 from eval.eval_smolinstruct import run_scoring
 
 save_json = args.save_json if args.save_json else str(pred_dir / "metrics.json")
 run_scoring(
 pred_dir,
 save_json=save_json,
 score_workers=args.score_workers,
 )
 print(f"\n[INFO] scoringcompleteresultsave: {save_json}")


if __name__ == "__main__":
 main()

