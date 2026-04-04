#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation Instruct/model SMolInstruct taskdatasamplegenerate compute_metrics.py readpredictionfile

- use data_dir prompt
- based on raw_data_dir template_dir generate prompt
 * datarandomfile <INPUT> replacedata input
 * --cot=Falsegenerate prompt " Please only output the answer."
- parameter
 * --raw_data_dir
 * --template_dir
 * --template_seed
 * --cot
"""

import argparse
import json
import os
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from functools import partial

from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ========== taskset compute_metrics.py alignment ==========
SMILES_TASKS = {
 "forward_synthesis",
 "retrosynthesis",
 "molecule_generation",
 "name_conversion-i2s",
}
NUMERIC_TASKS = {
 "property_prediction-esol",
 "property_prediction-lipo",
}
BOOLEAN_TASKS = {
 "property_prediction-bbbp",
 "property_prediction-clintox",
 "property_prediction-hiv",
 "property_prediction-sider",
}
SMILES_TASKS_MULTIMETRIC = {"retrosynthesis"}
TEXT_TASKS = {
 "molecule_captioning",
}
FORMULA_TASKS = {
 "name_conversion-i2f",
 "name_conversion-s2f",
 "name_conversion-s2i",
}


# ========== ==========
def is_target_jsonl(p: Path) -> bool:
 """recognitiontarget jsonl*.jsonl"""
 return p.is_file() and p.suffix.lower() == ".jsonl"


def task_name_from_filename(p: Path) -> str:
 """property_prediction-esol.jsonl -> property_prediction-esol"""
 return p.stem


def load_jsonl_rows(path: Path) -> List[Dict]:
 rows = []
 with path.open("r", encoding="utf-8") as f:
 for line in f:
 line = line.strip()
 if not line:
 continue
 rows.append(json.loads(line))
 return rows


def strip_code_fences(s: str) -> str:
 s = s.strip()
 if s.startswith("```") and s.endswith("```"):
 s = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", s)
 s = re.sub(r"\s*```$", "", s)
 s = s.strip()
 s = re.sub(r'^\s*(?:Answer\s*:|The answer is\s*: )\s*', "", s, flags=re.I)
 s = s.strip().strip('"').strip("'").strip()
 return s


def extract_number(text: str) -> str:
 if not text:
 return ""
 up = strip_code_fences(text)
 m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', up)
 return m.group(0) if m else ""

def extract_molecular_formula(text: str) -> str:
 """
 extractfirst“molecule”string C16H17N5 C24H18N2O3S2 …
 ifreturnsemptystring
 """
 # matchmode
 # - A-Z start a-z “Cl”, “Br” 
 # - 1 
 # - 
 # - length
 pattern = re.compile(r'\b(?:[A-Z][a-z]?[\d]*){2,}\b')
 match = pattern.search(text)
 if match:
 return match.group(0)
 return ""


def extract_yes_no(text: str) -> str:
 if not text:
 return ""
 t = strip_code_fences(text).lower()
 m = re.search(r'\b(yes|no)\b', t)
 if m:
 return m.group(1)
 m = re.search(r'\b(true|false)\b', t)
 if m:
 return "yes" if m.group(1) == "true" else "no"
 m = re.search(r'\b(y|n)\b', t)
 if m:
 return "yes" if m.group(1) == "y" else "no"
 return ""


_SMILES_CHARS = r"A-Za-z0-9@+\-\[\]\(\)=#\$\/\\\.\:%"


def extract_smiles(text: str) -> str:
 """generateextract SMILES string"""
 import re
 if not text:
 return ""
 s = strip_code_fences(text or "")
 lines = s.splitlines() if s else []

 # SMILES 
 pattern = re.compile(rf'([{_SMILES_CHARS}]+)')
 candidates = []
 for ln in lines:
 candidates.extend(pattern.findall(ln))
 if not candidates:
 return ""

 # filterlength 2contains SMILES feature
 def is_potential_smiles(tok: str) -> bool:
 if len(tok) < 2:
 return False
 has_digit = any(c.isdigit() for c in tok)
 has_special = any(c in "=#()[]@+/-\\" for c in tok)
 return has_digit or has_special

 filtered = [c for c in candidates if is_potential_smiles(c)]
 if not filtered:
 return ""

 #
 best = max(filtered, key=len)
 if best.endswith('.'):
 best = best[:-1]
 return best


def clean_text(text: str) -> str:
 s = strip_code_fences(text)
 return s.splitlines()[0].strip() if s else ""


def postprocess_by_task(task: str, raw_output: str) -> List[Optional[str]]:
 # if task in NUMERIC_TASKS:
 # num = extract_number(raw_output)
 # return [num if num is not None else ""]
 # if task in BOOLEAN_TASKS:
 # yn = extract_yes_no(raw_output)
 # return [yn if yn is not None else ""]
 # if task in SMILES_TASKS:
 # smi = extract_smiles(raw_output)
 # return [smi if smi is not None else ""]
 # if task in TEXT_TASKS:
 # return [clean_text(raw_output)]
 # if task in FORMULA_TASKS:
 # return [clean_text(raw_output)]
 # raise Exception(f"task {task} postprocess_by_task ")

 # list
 if task in NUMERIC_TASKS:
 num = extract_number(raw_output)
 return num
 if task in BOOLEAN_TASKS:
 yn = extract_yes_no(raw_output)
 return yn
 if task in SMILES_TASKS:
 smi = extract_smiles(raw_output)
 return smi
 if task in TEXT_TASKS:
 return clean_text(raw_output)
 if task in FORMULA_TASKS:
 if task == "name_conversion-s2f" or task == "name_conversion-i2f":
 return extract_molecular_formula(raw_output)
 else:
 return clean_text(raw_output)
 raise Exception(f"task {task} postprocess_by_task ")


# ========== inference ==========
def call_local_transformers_batch(
 tokenizer,
 model,
 prompts: List[Union[str, List[Dict[str, str]]]],
 is_chat: bool = False,
 max_new_tokens: int = 256,
 temperature: float = 0.0,
 top_p: float = 1.0,
):
 tokenizer.padding_side = "left"
 if tokenizer.pad_token_id is None:
 tokenizer.pad_token_id = tokenizer.eos_token_id

 if is_chat:
 rendered: List[str] = []
 for p in prompts:
 if isinstance(p, str):
 messages = [{"role": "user", "content": p}]
 else:
 messages = p
 rendered.append(
 tokenizer.apply_chat_template(
 messages,
 tokenize=False,
 add_generation_prompt=True,
 )
 )
 enc = tokenizer(rendered, return_tensors="pt", padding=True)
 else:
 raise Exception("currentsupports is_chat=True model")

 inputs = {k: v.to(model.device) for k, v in enc.items()}

 do_sample = temperature > 0
 generate_kwargs = {
 "max_new_tokens": max_new_tokens,
 "do_sample": do_sample,
 "pad_token_id": tokenizer.pad_token_id,
 "eos_token_id": getattr(tokenizer, "eos_token_id", None),
 "repetition_penalty": 1.06,
 "no_repeat_ngram_size": 3,
 }
 if do_sample:
 generate_kwargs.update({
 "temperature": temperature,
 "top_p": top_p,
 })

 with torch.inference_mode():
 output_ids = model.generate(**inputs, **generate_kwargs)

 outs = tokenizer.batch_decode(output_ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
 return [o.strip() for o in outs], output_ids


# ========== based onbuild prompt ==========
def build_prompts_from_template(raw_rows: List[Dict], template_list: List[Dict], cot: bool) -> List[str]:
 prompts = []
 for row in raw_rows:
 template = random.choice(template_list)
 # supportsdataformat
 # 1. SMolInstruct format "input" 
 # 2. messages format messages extract user content
 if "input" in row:
 input_str = row.get("input", "")
 elif "messages" in row:
 # messages formatextract user 
 messages = row.get("messages", [])
 user_msg = next((msg.get("content", "") for msg in messages if msg.get("role") == "user"), "")
 input_str = user_msg
 else:
 input_str = ""
 
 prompt = template["input"].replace("<INPUT>", input_str)
 # OUTPUTi.e.
 prompt = prompt.replace("<OUTPUT>", "")
 if not cot:
 # useformatdescription
 prompt = prompt.strip() + "\n\nPlease only output the answer without any explanation or additional text."
 prompts.append(prompt)
 return prompts


# ========== pipeline ==========
def evaluate_file(
 call_model_func,
 jsonl_path: Path,
 template_path: Path,
 output_dir: Path,
 is_chat_model: bool,
 cot: bool,
 verbose_every: int = 50,
 data_limit: int = 0,
 full_prompts_getter=None, # optionalfunctionforgetpromptlistcontainsfew-shot prefix
 tokenizer=None, # optionaltokenizerfordynamicgetspecial tokensassistantmark
) -> Tuple[str, int]:
 task = task_name_from_filename(jsonl_path)
 rows = load_jsonl_rows(jsonl_path)
 if data_limit > 0:
 rows = rows[:data_limit]
 print(f"[INFO] Evaluating {task}, {len(rows)} samples from {jsonl_path}")

 template_list = json.loads(template_path.read_text(encoding="utf-8"))
 out_path = output_dir / f"{task}.jsonl"
 out_path.parent.mkdir(parents=True, exist_ok=True)

 n = 0
 batch_size = 32

 with out_path.open("w", encoding="utf-8") as fw:
 for start in tqdm(range(0, len(rows), batch_size), desc=f"Evaluating {task}"):
 end = min(start + batch_size, len(rows))
 batch = rows[start:end]
 prompts = build_prompts_from_template(batch, template_list, cot=cot)
 outputs, raw_outputs_full = call_model_func(prompts=prompts)
 
 # ifreturnsoriginaloutputuseotherwiseuseprocessoutput
 if raw_outputs_full is not None and len(raw_outputs_full) == len(outputs):
 raw_outputs = raw_outputs_full
 else:
 # ifreturnsoriginaloutputuseprocessoutput
 raw_outputs = outputs
 
 # getpromptlistifgetterfunctionfew-shot
 full_prompts = None
 if full_prompts_getter is not None:
 full_prompts = full_prompts_getter()
 # lengthmatch
 if full_prompts and len(full_prompts) == len(raw_outputs):
 prompts_to_use = full_prompts
 else:
 prompts_to_use = prompts
 else:
 prompts_to_use = prompts

 for local_idx, (row, raw_output, prompt) in enumerate(zip(batch, raw_outputs, prompts_to_use), start=1):
 # ✅ processraw_output generate returns
 # ✅ answer_only raw_output samei.e. raw_answer
 answer_only = raw_output

 # answer_only extract predfor raw_output/answer_only 
 try:
 # tasktypedefine
 try:
 from eval.eval_smolinstruct import TEXT_TASKS, SMILES_TASKS, FORMULA_ELEMENT_TASKS, FORMULA_SPLIT_TASKS, NUMBER_TASKS, BOOLEAN_TASKS
 except ImportError:
 # iffailusedefaulttaskset
 TEXT_TASKS = {"molecule_captioning"}
 SMILES_TASKS = {"forward_synthesis", "retrosynthesis", "molecule_generation", "name_conversion-i2s"}
 FORMULA_ELEMENT_TASKS = {"name_conversion-i2f", "name_conversion-s2f"}
 FORMULA_SPLIT_TASKS = {"name_conversion-s2i"}
 NUMBER_TASKS = {"property_prediction-esol", "property_prediction-lipo"}
 BOOLEAN_TASKS = {"property_prediction-bbbp", "property_prediction-clintox", "property_prediction-hiv", "property_prediction-sider"}
 
 from eval.extract_prediction import extract_prediction_from_raw
 # useanswer_onlyextractpred
 pred = extract_prediction_from_raw(
 raw_output=None, # useraw_output
 task_name=task,
 answer_only=answer_only, # useanswer_only
 text_tasks=TEXT_TASKS,
 smiles_tasks=SMILES_TASKS,
 formula_element_tasks=FORMULA_ELEMENT_TASKS,
 formula_split_tasks=FORMULA_SPLIT_TASKS,
 number_tasks=NUMBER_TASKS,
 boolean_tasks=BOOLEAN_TASKS,
 )
 except (ImportError, Exception) as e:
 # ifextractfailusepostprocess
 pred = postprocess_by_task(task, raw_output)
 
 # supportsdataformat
 # 1. SMolInstruct format "gold" "input" 
 # 2. messages format messages extract assistant content gold
 gold = row.get("gold") or row.get("output")
 if not gold and "messages" in row:
 # messages formatextract assistant gold
 messages = row.get("messages", [])
 gold = next((msg.get("content", "") for msg in messages if msg.get("role") == "assistant"), "")
 
 input_str = row.get("input", "")
 if not input_str and "messages" in row:
 # messages formatextract user input
 messages = row.get("messages", [])
 input_str = next((msg.get("content", "") for msg in messages if msg.get("role") == "user"), "")
 
 item_out = {
 "prompt": prompt, # promptalreadypromptcontainsfew-shot prefixsuffix
 "gold": gold,
 "pred": pred,
 "input": input_str,
 "raw_output": raw_output, # originaloutput
 "answer_only": answer_only, # answerversionassistantextractremovethink
 "sample_id": row.get("sample_id"),
 "task": task,
 }
 # print(f"------------")
 # print(f"prompt: {prompt}")
 # print(f"------------")
 # print(f"raw_output: {raw_output}")
 # print(f"------------")
 # print(f"pred: {pred}")
 if "target" in row:
 item_out["target"] = row.get("target")
 fw.write(json.dumps(item_out, ensure_ascii=False) + "\n")
 n += 1

 global_idx = start + local_idx
 # if verbose_every and (global_idx) % verbose_every == 0:
 # print(f"[{task}] {global_idx} samples processed. Example:")
 # print("PROMPT:", (prompt[:300] + "...") if len(prompt) > 300 else prompt)
 # print("RAW :", (raw_output[:300] + "...") if len(raw_output) > 300 else raw_output)
 # print("PRED :", pred)

 print(f"[DONE] {task}: wrote {n} lines -> {out_path}")
 return task, n


def main():
 parser = argparse.ArgumentParser(description="Evaluate models on SMolInstruct raw_data + template to generate prompts and predictions for compute_metrics.py.")
 parser.add_argument("--raw_data_dir", type=str, required=True, help="originaldatadirectorycontains *.jsonl")
 parser.add_argument("--template_dir", type=str, required=True, help="directorycontains *.json")
 parser.add_argument("--data_limit", type=int, default=0)
 parser.add_argument("--output_dir", type=str, default="predictions_smol")
 parser.add_argument("--model", type=str, required=True)
 parser.add_argument("--dtype", type=str, default="auto",
 choices=["auto", "float16", "bfloat16", "float32"])
 parser.add_argument("--device", type=str, default="auto",
 choices=["auto", "cuda", "cpu"])
 parser.add_argument("--max_new_tokens", type=int, default=128)
 parser.add_argument("--temperature", type=float, default=0.0)
 parser.add_argument("--top_p", type=float, default=1.0)
 parser.add_argument("--sampling_seed", type=int, default=42)
 parser.add_argument("--template_seed", type=int, default=42)
 parser.add_argument("--cot", action="store_true", help="whether chain-of-thought 'Please only output the answer.'")
 parser.add_argument("--verbose_every", type=int, default=50)
 args = parser.parse_args()

 if args.device == "auto":
 device = "cuda" if torch.cuda.is_available() else "cpu"
 else:
 device = args.device
 if args.dtype == "auto":
 torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
 else:
 torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

 print(f"[INFO] Loading model: {args.model} on {device} (dtype={torch_dtype})")

 if args.data_limit < 0:
 raise ValueError("--data_limit ")

 model = AutoModelForCausalLM.from_pretrained(
 args.model,
 torch_dtype=torch_dtype,
 trust_remote_code=True,
 device_map="auto" if device == "cuda" else None,
 ).to(device)
 tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

 if tokenizer.pad_token is None:
 assert tokenizer.eos_token is not None
 tokenizer.pad_token = tokenizer.eos_token

 if hasattr(model, "config"):
 model.config.use_cache = True
 if hasattr(model, "generation_config"):
 try:
 model.generation_config.use_cache = True
 except Exception:
 pass
 model.eval()

 is_chat = "instruct" in args.model.lower() or "chat" in args.model.lower()
 print(f"[INFO] Is Chat Model: {is_chat}")

 if args.temperature > 0:
 torch.manual_seed(int(args.sampling_seed))
 random.seed(args.template_seed)

 sampling_params = {
 "temperature": args.temperature,
 "top_p": args.top_p,
 "max_new_tokens": args.max_new_tokens,
 }
 call_model = partial(
 call_local_transformers_batch,
 tokenizer=tokenizer,
 model=model,
 is_chat=is_chat,
 **sampling_params,
 )

 raw_data_dir = Path(args.raw_data_dir).expanduser().resolve()
 template_dir = Path(args.template_dir).expanduser().resolve()
 output_dir = Path(args.output_dir).expanduser().resolve()
 files = sorted([p for p in raw_data_dir.iterdir() if is_target_jsonl(p)])

 if not files:
 print(f"[WARN] directory *.jsonl {raw_data_dir}")
 return

 print(f"[INFO] evaluation {len(files)} taskfile")
 for p in files:
 print(f" - {p.name}")

 total = 0
 summary = []
 for jsonl_path in files:
 task = task_name_from_filename(jsonl_path)
 template_path = template_dir / f"{task}.json"
 if not template_path.exists():
 print(f"[WARN] {template_path}skiptask")
 continue

 task_name, n = evaluate_file(
 call_model_func=call_model,
 jsonl_path=jsonl_path,
 template_path=template_path,
 output_dir=output_dir,
 is_chat_model=is_chat,
 cot=args.cot,
 verbose_every=args.verbose_every,
 data_limit=args.data_limit,
 )
 summary.append({"task": task_name, "num_samples": n, "outfile": str(output_dir / f"{task_name}.jsonl")})
 total += n

 summary_path = output_dir / "eval_summary.json"
 with summary_path.open("w", encoding="utf-8") as f:
 json.dump({
 "model": args.model,
 "gen_config": sampling_params,
 "raw_data_dir": str(raw_data_dir),
 "template_dir": str(template_dir),
 "output_dir": str(output_dir),
 "total_samples": total,
 "per_task": summary,
 }, f, ensure_ascii=False, indent=2)
 print(f"\n[INFO] totalfilewrite{summary_path}")
 print("[INFO] runpython compute_metrics.py --prediction_dir", str(output_dir))


if __name__ == "__main__":
 main()