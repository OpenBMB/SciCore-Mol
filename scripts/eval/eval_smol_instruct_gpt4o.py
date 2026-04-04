#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 OpenAI-compatible ChatCompletions APIe.g. OpenAI-compatible APIrun SMolInstruct
- datapath/path eval_smolinstruct_batch.py raw_data_dir + template_dir
- outputeach task <task>.jsonlconsistentprompt/gold/pred/input/raw_output/answer_only/sample_id/task
- pred extract eval.extract_prediction.extract_prediction_from_raw
- supportstaskevaluation + progress + optional include_tasks
"""

import os
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import requests
from tqdm import tqdm

# -----------------------
# tasktype eval_smolinstruct.py / extract_prediction alignment
# -----------------------
TEXT_TASKS = {"molecule_captioning"}
SMILES_TASKS = {"forward_synthesis", "retrosynthesis", "molecule_generation", "name_conversion-i2s"}
FORMULA_ELEMENT_TASKS = {"name_conversion-i2f", "name_conversion-s2f"}
FORMULA_SPLIT_TASKS = {"name_conversion-s2i"}
NUMBER_TASKS = {"property_prediction-esol", "property_prediction-lipo"}
BOOLEAN_TASKS = {"property_prediction-bbbp", "property_prediction-clintox", "property_prediction-hiv", "property_prediction-sider"}

# -----------------------
# batch evaluatordirectory
# -----------------------
from importlib.machinery import SourceFileLoader
from importlib.util import spec_from_loader, module_from_spec

def import_module_from_path(name: str, path: str):
 loader = SourceFileLoader(name, path)
 spec = spec_from_loader(loader.name, loader)
 mod = module_from_spec(spec)
 loader.exec_module(mod)
 return mod

# evaluator pathalready
EVALUATOR_PATH = "${SCICORE_ROOT:-/path/to/scicore-mol}/eval/eval_smolinstruct_batch.py"
evaluator = import_module_from_path("eval_smolinstruct_batch", EVALUATOR_PATH)



def _task_suffix(task: str) -> str:
 """
 GPT constraintoutputformat 'Okay'/ pred=null
 """
 task = str(task)
 if task in SMILES_TASKS:
 return "\n\nOutput MUST be exactly one SMILES wrapped by <SMILES> and </SMILES>. No other text."
 if task in NUMBER_TASKS:
 return "\n\nOutput MUST be exactly one number wrapped by <NUMBER> and </NUMBER>. No other text."
 if task in BOOLEAN_TASKS:
 return "\n\nOutput MUST be exactly True or False wrapped by <BOOLEAN> and </BOOLEAN>. No other text."
 if task in FORMULA_ELEMENT_TASKS:
 return "\n\nOutput MUST be a chemical formula wrapped by <FORMULA> and </FORMULA>. No other text."
 if task in FORMULA_SPLIT_TASKS:
 return "\n\nOutput MUST be a tokenized formula wrapped by <FORMULA> and </FORMULA>. No other text."
 if task in TEXT_TASKS:
 return "\n\nOutput ONLY the final answer text. No extra commentary."
 # defaultallow
 return "\n\nOutput ONLY the answer. No explanation."


def call_chat_completions(
 base_url: str,
 api_key: str,
 model: str,
 prompts: List[str],
 temperature: float,
 max_tokens: int,
 timeout: int,
 max_retries: int,
 task: str,
) -> List[str]:
 """
 serialrequestif max_workers threadi.e.
 “run”defaultserialretry
 """
 url = f"{base_url.rstrip('/')}/chat/completions"
 headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

 system = (
 "You are a strict chemistry benchmark solver.\n"
 "Follow the instruction EXACTLY.\n"
 "Return ONLY the required answer format.\n"
 "Never say 'OK', never add explanations.\n"
 )

 outs: List[str] = []
 for p in prompts:
 payload = {
 "model": model,
 "messages": [
 {"role": "system", "content": system},
 {"role": "user", "content": p + _task_suffix(task)},
 ],
 "temperature": float(temperature),
 "max_tokens": int(max_tokens),
 }

 text = ""
 for a in range(1, max_retries + 1):
 try:
 r = requests.post(url, headers=headers, json=payload, timeout=timeout)
 r.raise_for_status()
 j = r.json()
 text = (j["choices"][0]["message"]["content"] or "").strip()
 break
 except Exception:
 if a == max_retries:
 text = ""
 break
 time.sleep((2 ** (a - 1)) + random.random())
 outs.append(text)
 return outs


def build_call_model_api(args):
 """
 evaluator.evaluate_file call_model_func(prompts=...) -> (outputs, raw_outputs_full)
 outputs/raw_outputs_full GPT originalreturns eval_smolinstruct_batch.py 
 """
 api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
 if not api_key:
 raise RuntimeError("Missing API key. Provide --api_key or set env OPENAI_API_KEY")

 base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

 def _call_model(prompts: List[str], task: str = ""):
 raw = call_chat_completions(
 base_url=base_url,
 api_key=api_key,
 model=args.model,
 prompts=prompts,
 temperature=args.temperature,
 max_tokens=args.max_tokens,
 timeout=args.timeout,
 max_retries=args.max_retries,
 task=task,
 )
 # evaluator raw_output write raw_output/answer_only extract_prediction
 return raw, raw

 return _call_model


def main():
 parser = argparse.ArgumentParser()
 parser.add_argument("--raw_data_dir", type=str, required=True, help="construct SMolInstruct jsonl directorycontains *.jsonl")
 parser.add_argument("--template_dir", type=str, required=True, help="directorycontains <task>.json")
 parser.add_argument("--out_dir", type=str, required=True)

 # API
 parser.add_argument("--model", type=str, default="gpt-4o")
 parser.add_argument("--base_url", type=str, default="${API_BASE:-https://api.openai.com/v1}")
 parser.add_argument("--api_key", type=str, default="")
 parser.add_argument("--temperature", type=float, default=0.0)
 parser.add_argument("--max_tokens", type=int, default=128)
 parser.add_argument("--timeout", type=int, default=120)
 parser.add_argument("--max_retries", type=int, default=6)

 # eval behavior
 parser.add_argument("--data_limit", type=int, default=0)
 parser.add_argument("--cot", action="store_true", help="append 'Please only output ...'")
 parser.add_argument("--verbose_every", type=int, default=0)
 parser.add_argument("--include_tasks", type=str, default="", help="runtask")

 args = parser.parse_args()

 raw_data_dir = Path(args.raw_data_dir).expanduser().resolve()
 template_dir = Path(args.template_dir).expanduser().resolve()
 out_dir = Path(args.out_dir).expanduser().resolve()
 out_dir.mkdir(parents=True, exist_ok=True)

 # build API callfunction
 call_model_func = build_call_model_api(args)

 # taskfile evaluator.is_target_jsonl
 files = sorted([p for p in raw_data_dir.iterdir() if evaluator.is_target_jsonl(p)])
 if args.include_tasks.strip():
 allow = {t.strip() for t in args.include_tasks.split(",") if t.strip()}
 files = [p for p in files if evaluator.task_name_from_filename(p) in allow]

 if not files:
 raise RuntimeError(f"No task jsonl found under {raw_data_dir}")

 total = 0
 per_task = []

 print(f"[INFO] tasks = {len(files)}")
 for p in files:
 print(" -", p.name, "->", evaluator.task_name_from_filename(p))

 # task looptask internal batch tqdm
 for jsonl_path in tqdm(files, desc="SMolInstruct tasks", unit="task"):
 task = evaluator.task_name_from_filename(jsonl_path)
 template_path = template_dir / f"{task}.json"
 if not template_path.exists():
 print(f"[WARN] template missing: {template_path} (skip)")
 continue

 task_name, n = evaluator.evaluate_file(
 call_model_func=lambda prompts, _task=task: call_model_func(prompts, task=_task),
 jsonl_path=jsonl_path,
 template_path=template_path,
 output_dir=out_dir,
 is_chat_model=True,
 cot=args.cot,
 verbose_every=args.verbose_every if args.verbose_every > 0 else 10**18,
 data_limit=args.data_limit,
 )
 per_task.append({"task": task_name, "num_samples": n, "outfile": str(out_dir / f"{task_name}.jsonl")})
 total += n

 summary = {
 "model": args.model,
 "base_url": args.base_url,
 "raw_data_dir": str(raw_data_dir),
 "template_dir": str(template_dir),
 "out_dir": str(out_dir),
 "total_samples": total,
 "per_task": per_task,
 "gen": {
 "temperature": args.temperature,
 "max_tokens": args.max_tokens,
 "timeout": args.timeout,
 "max_retries": args.max_retries,
 },
 }
 (out_dir / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
 print(f"\n[DONE] wrote eval_summary.json, total={total}")
 print(f"[NEXT] run scoring (unchanged): python eval_smolinstruct.py --score_only 1 --output_dir {out_dir}")


if __name__ == "__main__":
 main()
