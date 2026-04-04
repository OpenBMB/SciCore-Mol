#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script - predictionresultscoring

 python scripts/score_only.py \
 --prediction_dir ${SCICORE_ROOT:-/path/to/scicore-mol}/test_output_eval_qwen_GNN_nofreeze_checkpoint-39_fewshot \
 --save_json ${SCICORE_ROOT:-/path/to/scicore-mol}/test_output_eval_qwen_GNN_nofreeze_checkpoint-39_fewshot/scored_results.json
"""

import argparse
import sys
from pathlib import Path

# directorypath
_project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_project_root))

from eval.eval_smolinstruct import run_scoring


def main():
 parser = argparse.ArgumentParser(description="predictionresultscoring")
 parser.add_argument(
 "--prediction_dir",
 type=str,
 default="",
 help="containspredictionresult jsonl filedirectorydefault: $SciCore-Mol_ROOT/eval_results/results",
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
 help="scoringuseprocess1=process>1=process",
 )
 parser.add_argument(
 "--skip_tasks",
 type=str,
 default="",
 help="skiptask",
 )
 
 args = parser.parse_args()
 
 # Default prediction root: SciCore-Mol_ROOT/eval_results/results
 import os
 scicore-mol_root = Path(os.environ.get("SciCore-Mol_ROOT", "${SCICORE_ROOT:-/path/to/scicore-mol}"))
 default_pred_root = scicore-mol_root / "eval_results" / "results"
 pred_dir = Path(args.prediction_dir).expanduser().resolve() if args.prediction_dir else default_pred_root
 if not pred_dir.exists():
 print(f"errordirectory: {pred_dir}")
 sys.exit(1)
 
 if not pred_dir.is_dir():
 print(f"errordirectory: {pred_dir}")
 sys.exit(1)
 
 # callscoringfunction
 save_json = args.save_json if args.save_json else str(pred_dir / "scored_results.json")
 run_scoring(
 pred_dir,
 save_json=save_json,
 score_workers=args.score_workers,
 skip_tasks=args.skip_tasks,
 )
 
 print(f"\n[INFO] scoringcompleteresultsave: {save_json}")


if __name__ == "__main__":
 main()

