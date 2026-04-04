#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ChemBench4K (AI4Chem/ChemBench4K) ALL benchmark tasks evaluation using GPT-4o API
with multi-threading support.

How:
1) List all files under split dir (default: test/) that end with "_benchmark.json"
 from HuggingFace repo.
2) For each benchmark json file:
 - Load dataset via datasets.load_dataset(data_files=...)
 - For each example, build MCQ prompt
 - Call GPT-4o API to get prediction (A/B/C/D)
 - Use multi-threading for concurrent API calls

Output:
- out_dir/pred_{benchmark_name}.jsonl
- out_dir/summary.json

Usage:
 python eval_chembench4k_gpt4o.py \
 --out_dir ${SCICORE_ROOT:-/path/to/scicore-mol}/eval_chembench_gpt4o \
 --split test \
 --max_workers 8 \
 --max_items -1
"""

from __future__ import annotations
import argparse
import json
import os
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import requests

try:
 from datasets import load_dataset # type: ignore
 HAS_DATASETS = True
except Exception:
 HAS_DATASETS = False

try:
 from huggingface_hub import HfApi # type: ignore
 HAS_HF_HUB = True
except Exception:
 HAS_HF_HUB = False

# API configkey 
OPENAI_API_KEY = "${OPENAI_API_KEY}"
OPENAI_BASE_URL = "${API_BASE:-https://api.openai.com/v1}"

DATASET_REPO = "AI4Chem/ChemBench4K"
CHOICES = ["A", "B", "C", "D"]


def list_benchmark_files(split: str, max_retries: int = 3) -> List[str]:
 """
 List all benchmark json files under split/ (e.g., 'test/') in the HF dataset repo.
 """
 if not HAS_HF_HUB:
 raise RuntimeError("huggingface_hub not installed. Please: pip install -U huggingface_hub")
 
 # useif
 original_endpoint = os.environ.get("HF_ENDPOINT")
 
 for attempt in range(max_retries):
 try:
 # iffail
 if attempt > 0:
 print(f"[ {attempt + 1}/{max_retries}] HuggingFace ...")
 os.environ.pop("HF_ENDPOINT", None) # removeset
 else:
 # usecurrentset
 pass
 
 api = HfApi()
 files = api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset")
 prefix = f"{split}/"
 bench = []
 for f in files:
 if f.startswith(prefix) and f.endswith("_benchmark.json"):
 bench.append(f)
 bench.sort()
 if not bench:
 raise RuntimeError(f"No *_benchmark.json found under {split}/ in {DATASET_REPO}")
 
 # restoreoriginalset
 if original_endpoint:
 os.environ["HF_ENDPOINT"] = original_endpoint
 elif "HF_ENDPOINT" in os.environ:
 os.environ.pop("HF_ENDPOINT")
 
 return bench
 except Exception as e:
 if attempt < max_retries - 1:
 wait_time = (2 ** attempt) + random.random()
 print(f"[error] {type(e).__name__}: {e}")
 print(f"[wait {wait_time:.1f} retry...]")
 time.sleep(wait_time)
 else:
 # restoreoriginalset
 if original_endpoint:
 os.environ["HF_ENDPOINT"] = original_endpoint
 elif "HF_ENDPOINT" in os.environ:
 os.environ.pop("HF_ENDPOINT")
 raise RuntimeError(f"fileretry {max_retries} : {e}")
 
 # should
 raise RuntimeError("error")


def load_task_hf(file_path_in_repo: str, split: str, max_retries: int = 3) -> List[Dict[str, Any]]:
 """
 file_path_in_repo looks like: 'test/Name_Conversion_benchmark.json'
 split should be 'test' to match.
 """
 if not HAS_DATASETS:
 raise RuntimeError("datasets not installed. Please: pip install -U datasets")
 
 # useif
 original_endpoint = os.environ.get("HF_ENDPOINT")
 
 for attempt in range(max_retries):
 try:
 # iffail
 if attempt > 0:
 print(f"[ {attempt + 1}/{max_retries}] HuggingFace loaddata...")
 os.environ.pop("HF_ENDPOINT", None) # removeset
 
 # datasets expects data_files as mapping split->file
 data_files = {split: file_path_in_repo}
 ds = load_dataset(DATASET_REPO, data_files=data_files, split=split)
 
 # restoreoriginalset
 if original_endpoint:
 os.environ["HF_ENDPOINT"] = original_endpoint
 elif "HF_ENDPOINT" in os.environ:
 os.environ.pop("HF_ENDPOINT")
 
 return [dict(x) for x in ds]
 except Exception as e:
 if attempt < max_retries - 1:
 wait_time = (2 ** attempt) + random.random()
 print(f"[error] {type(e).__name__}: {e}")
 print(f"[wait {wait_time:.1f} retry...]")
 time.sleep(wait_time)
 else:
 # restoreoriginalset
 if original_endpoint:
 os.environ["HF_ENDPOINT"] = original_endpoint
 elif "HF_ENDPOINT" in os.environ:
 os.environ.pop("HF_ENDPOINT")
 raise RuntimeError(f"loaddataretry {max_retries} : {e}")
 
 # should
 raise RuntimeError("error")


def format_prompt_mcq(ex: Dict[str, Any], task_name: str) -> str:
 """
 ChemBench4K benchmark json format: has 'question', 'A','B','C','D','answer'
 We build a strict MCQ prompt for GPT-4o.
 """
 q = (ex.get("question") or "").strip()
 a = (ex.get("A") or "").strip()
 b = (ex.get("B") or "").strip()
 c = (ex.get("C") or "").strip()
 d = (ex.get("D") or "").strip()

 return (
 "You are taking a multiple-choice chemistry benchmark.\n"
 "Choose the single best option.\n"
 "You MUST output ONLY one letter among A, B, C, D.\n"
 "No explanation. No punctuation. No extra text.\n\n"
 f"Benchmark: {task_name}\n\n"
 f"Question:\n{q}\n\n"
 "Options:\n"
 f"A. {a}\n"
 f"B. {b}\n"
 f"C. {c}\n"
 f"D. {d}\n\n"
 "Answer (A/B/C/D):"
 )


def extract_choice_from_text(text: str) -> str:
 """
 GPT-4o outputextract A/B/C/D
 """
 text = text.strip().upper()
 # match
 for c in CHOICES:
 if text.startswith(c) or f" {c}" in text or f"\n{c}" in text:
 return c
 # ifreturnsfirstifvalid
 if text and text[0] in CHOICES:
 return text[0]
 return "A" # defaultreturns A


def call_gpt4o(
 api_key: str,
 base_url: str,
 prompt: str,
 max_retries: int = 5,
 model: str = "gpt-4o",
) -> str:
 """
 call GPT-4o APIuse requests callretry
 """
 system_prompt = (
 "You are a chemistry expert taking a multiple-choice test. "
 "You must output ONLY a single letter: A, B, C, or D. "
 "No explanation, no punctuation, no extra text."
 )
 
 url = f"{base_url}/chat/completions"
 headers = {
 "Authorization": f"Bearer {api_key}",
 "Content-Type": "application/json",
 }
 payload = {
 "model": model,
 "messages": [
 {"role": "system", "content": system_prompt},
 {"role": "user", "content": prompt},
 ],
 "max_tokens": 10, # needs
 "temperature": 0.0, # output
 }
 
 for attempt in range(1, max_retries + 1):
 try:
 response = requests.post(url, headers=headers, json=payload, timeout=30)
 response.raise_for_status()
 data = response.json()
 result = (data["choices"][0]["message"]["content"] or "").strip()
 return extract_choice_from_text(result)
 except Exception as e:
 # + 
 if attempt < max_retries:
 time.sleep((2 ** (attempt - 1)) + random.random())
 else:
 print(f"[ERROR after {max_retries} attempts] {e}")
 return "A" # defaultreturns A
 return "A"


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--out_dir", type=str, required=True)
 
 ap.add_argument("--split", type=str, default="test", choices=["test", "train", "validation"])
 ap.add_argument("--use_hf", type=int, default=1, help="must be 1 (HF datasets)")
 ap.add_argument("--max_items", type=int, default=-1, help="-1 means all items per benchmark file")
 ap.add_argument("--only_files", type=str, default="", help="comma-separated exact file names (e.g., Name_Conversion_benchmark.json)")
 ap.add_argument("--max_workers", type=int, default=8, help="number of concurrent threads for API calls")
 ap.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model name")
 ap.add_argument("--max_retries", type=int, default=5, help="max retries for API calls")
 args = ap.parse_args()

 out_dir = Path(args.out_dir)
 out_dir.mkdir(parents=True, exist_ok=True)

 if args.use_hf != 1:
 raise ValueError("This script is designed for HF loading only. Please set --use_hf 1")

 # setrandomforretry
 random.seed(42)

 # useenvironmentvariabledefaultvalue
 api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
 base_url = os.environ.get("OPENAI_BASE_URL", OPENAI_BASE_URL)

 # List benchmark files
 bench_files_full = list_benchmark_files(args.split) # like ['test/XXX_benchmark.json', ...]
 
 # defaultprocess3benchmark
 default_benchmarks = {
 "Product_Prediction_benchmark.json",
 "Retrosynthesis_benchmark.json",
 "Yield_Prediction_benchmark.json",
 }
 
 if args.only_files.strip():
 only = {x.strip() for x in args.only_files.split(",") if x.strip()}
 bench_files_full = [f for f in bench_files_full if f.split("/")[-1] in only]
 bench_files_full.sort()
 if not bench_files_full:
 raise RuntimeError(f"--only_files provided but matched nothing. only={sorted(list(only))}")
 else:
 # if --only_filesdefaultprocess3benchmark
 bench_files_full = [f for f in bench_files_full if f.split("/")[-1] in default_benchmarks]
 bench_files_full.sort()
 if bench_files_full:
 print(f"[INFO] defaultprocess3benchmark: {[f.split('/')[-1] for f in bench_files_full]}")
 else:
 print(f"[WARNING] default3benchmarkprocessallbenchmarkfile")

 summary: Dict[str, Any] = {}
 overall_total = 0
 overall_correct = 0

 pbar_tasks = tqdm(bench_files_full, desc="Benchmarks", unit="file")
 for file_path in pbar_tasks:
 bench_name = file_path.split("/")[-1].replace(".json", "") # e.g. Name_Conversion_benchmark
 exs = load_task_hf(file_path_in_repo=file_path, split=args.split)
 if args.max_items is not None and args.max_items > 0:
 exs = exs[:args.max_items]

 out_jsonl = out_dir / f"pred_{bench_name}.jsonl"
 
 total = 0
 correct = 0
 
 # tasklistprocessvalidsample
 jobs = []
 valid_indices = [] # recordvalidsampleoriginalindex
 for i, ex in enumerate(exs):
 gold = (ex.get("answer") or "").strip().upper()
 # Skip malformed
 if gold not in CHOICES:
 continue
 
 prompt = format_prompt_mcq(ex, bench_name)
 job_idx = len(jobs) # taskindex0start
 jobs.append({
 "job_idx": job_idx,
 "orig_idx": i, # originalindex
 "ex": ex,
 "gold": gold,
 "prompt": prompt,
 "bench_name": bench_name,
 "file_path": file_path,
 })
 valid_indices.append(i)
 
 # threadcall GPT-4o
 results_buffer: Dict[int, Dict[str, Any]] = {}
 
 def process_job(job: Dict[str, Any]) -> Dict[str, Any]:
 """processtask"""
 pred = call_gpt4o(api_key, base_url, job["prompt"], max_retries=args.max_retries, model=args.model)
 ok = (pred == job["gold"])
 return {
 "orig_idx": job["orig_idx"], # useoriginalindex
 "benchmark": job["bench_name"],
 "file": job["file_path"],
 "gold": job["gold"],
 "pred": pred,
 "correct": ok,
 }
 
 # execute
 if jobs:
 with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
 future_to_job = {executor.submit(process_job, job): job for job in jobs}
 for future in tqdm(as_completed(future_to_job), total=len(future_to_job), desc=f"{bench_name}", leave=False, unit="ex"):
 job = future_to_job[future]
 try:
 result = future.result()
 results_buffer[result["orig_idx"]] = result
 total += 1
 if result["correct"]:
 correct += 1
 except Exception as e:
 print(f"[ERROR] Job {job['orig_idx']} failed: {e}")
 results_buffer[job["orig_idx"]] = {
 "orig_idx": job["orig_idx"],
 "benchmark": job["bench_name"],
 "file": job["file_path"],
 "gold": job["gold"],
 "pred": "A", # default
 "correct": False,
 "error": str(e),
 }
 total += 1
 
 # originalwriteresult
 with out_jsonl.open("w", encoding="utf-8") as f:
 for orig_idx in sorted(results_buffer.keys()):
 rec = results_buffer[orig_idx]
 # idx 
 rec["idx"] = orig_idx
 f.write(json.dumps(rec, ensure_ascii=False) + "\n")

 acc = correct / total if total > 0 else 0.0
 summary[bench_name] = {
 "file": file_path,
 "acc": acc,
 "correct": correct,
 "total": total,
 }

 overall_total += total
 overall_correct += correct
 pbar_tasks.set_postfix(overall_acc=f"{(overall_correct/overall_total) if overall_total else 0.0:.3f}")

 summary["overall"] = {
 "acc": (overall_correct / overall_total) if overall_total else 0.0,
 "correct": overall_correct,
 "total": overall_total,
 "num_benchmarks": len(bench_files_full),
 "split": args.split,
 }

 (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
 print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
 main()
