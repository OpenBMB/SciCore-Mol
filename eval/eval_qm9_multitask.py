#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QM9 taskcomparisonevaluationscriptBaseline LLM vs MolAware+GNN

example
 CUDA_VISIBLE_DEVICES=4 python eval_qm9_multitask.py \
 --cfg configs/epoch2_config_modified.yaml \
 --ckpt_dir ${DATA_DIR:-/path/to/data}/MSMLM/model/llama3.2-chem-sft-gnn/1125_llm_gnn_loss/epoch1_random_gnn_freeze_llm \
 --test_file ${DATA_DIR:-/path/to/data}/MoleculeNet/qa_data/mol_qa_test.jsonl \
 --max_samples 1000 \
 --out_dir ${DATA_DIR:-/path/to/data}/MSMLM/model/llama3.2-chem-sft-gnn/1123_llm_gnn_loss/epoch2/eval_qm9_compare \
 --temperature 0.0
"""

import os
import re
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM
from datasets import load_dataset
import yaml

# allow numpy.dtype deserialize train_sft.py consistent
import numpy
if hasattr(torch.serialization, "add_safe_globals"):
 torch.serialization.add_safe_globals([numpy.dtype])

# module train_sft.py 
from modules.model_init import (
 init_tokenizer,
 init_llm,
 init_model,
)
from modules.mol_aware_lm import MolAwareCausalLM


QM9_TASKS = ["mu", "alpha", "homo", "lumo", "gap"]


# ------------------------- function ------------------------- #

def set_seed(seed: int):
 import random
 random.seed(seed)
 np.random.seed(seed)
 torch.manual_seed(seed)
 if torch.cuda.is_available():
 torch.cuda.manual_seed_all(seed)


def build_prompt_from_tokenizer(user_text: str, tokenizer) -> str:
 """
 according totokenizerinferpromptformatsupportsLlamaMistralQwenmodel
 """
 vocab = tokenizer.get_vocab()
 
 # 2. Qwen formatuse apply_chat_template
 if hasattr(tokenizer, "apply_chat_template"):
 try:
 formatted = tokenizer.apply_chat_template(
 [
 {"role": "system", "content": "You are a helpful chemist."},
 {"role": "user", "content": user_text},
 ],
 tokenize=False,
 add_generation_prompt=True,
 )
 return formatted + " "
 except Exception:
 pass
 
 # 1. Llama 3.x formatcontains header tokens
 if "<|start_header_id|>" in vocab and "<|end_header_id|>" in vocab:
 return (
 "<|start_header_id|>user<|end_header_id|>\n\n"
 f"{user_text}<|eot_id|>\n"
 "<|start_header_id|>assistant<|end_header_id|>\n\n"
 )
 
 # 3. Mistral formatuse [INST] label
 if "[INST]" in vocab or "</s>" in vocab:
 return f"[INST] {user_text} [/INST]"
 
 # 4. format
 return f"User: {user_text}\n\nAssistant: "


def build_llama32_prompt(user_text: str) -> str:
 """
 interfaceuse build_prompt_from_tokenizer
 """
 return (
 "<|start_header_id|>user<|end_header_id|>\n\n"
 f"{user_text}<|eot_id|>\n"
 "<|start_header_id|>assistant<|end_header_id|>\n\n"
 )


def extract_first_float(text: str) -> Optional[float]:
 """
 modelgenerate caption firstincluding
 """
 if not isinstance(text, str):
 return None
 matches = re.findall(
 r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?",
 text
 )
 if not matches:
 return None
 for m in matches:
 try:
 return float(m)
 except Exception:
 continue
 return None


def compute_regression_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
 """
 computeregressionmetricMAE / RMSE
 """
 yt = np.array(y_true, dtype=np.float64)
 yp = np.array(y_pred, dtype=np.float64)
 mae = float(np.mean(np.abs(yp - yt)))
 rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
 return {"mae": mae, "rmse": rmse, "n": int(len(yt))}


# ------------------------- modelload ------------------------- #

def load_molaware_from_ckpt(cfg: Dict[str, Any], ckpt_dir: str, device: str):
 """
 1) config initialize tokenizer + base LLM + MolAware 
 2) ckpt_dir load Stage2 trainingweightllm/ + extras/

 ckpt_dir e.g.: 
 ${CHECKPOINT_DIR:-/path/to/checkpoints}/epoch2/checkpoint-4000
 should:
 llm/ (HF Llama model)
 extras/gvp_encoder.pt
 extras/mol_adapter.pt
 extras/diffusion_adapter.pt (optional)
 """
 bf16 = bool(cfg["train"].get("bf16", False))
 mol_token = cfg["tokens"]["mol_token"]
 llm_name_or_path = cfg["paths"]["llm_name_or_path"] # define
 
 # checkpointllmdirectoryloadtokenizermodelmatch
 llm_dir = os.path.join(ckpt_dir, "llm")
 if os.path.isdir(llm_dir):
 print(f"[Eval] Loading tokenizer from checkpoint LLM: {llm_dir}")
 from transformers import AutoTokenizer
 tokenizer = AutoTokenizer.from_pretrained(llm_dir, trust_remote_code=True)
 # iftokenizermol_tokenneeds
 if mol_token not in tokenizer.get_vocab():
 tokenizer.add_tokens([mol_token], special_tokens=True)
 else:
 # configbase LLM
 print(f"[Eval] Initializing tokenizer from base LLM: {llm_name_or_path}")
 tokenizer = init_tokenizer(llm_name_or_path, mol_token)

 # set pad_token
 if tokenizer.pad_token is None:
 print(f"[Eval] Tokenizer.pad_token is None, set to eos_token")
 tokenizer.pad_token = tokenizer.eos_token
 tokenizer.pad_token_id = tokenizer.eos_token_id
 tokenizer.padding_side = "right"
 tokenizer.model_max_length = int(cfg["train"]["max_seq_length"])

 # ------------ ckpt_dir/llm load LLMmatch ------------
 # llm_dir alreadydefine
 if os.path.isdir(llm_dir):
 print(f"[Eval] Loading fine-tuned LLM from checkpoint: {llm_dir}")
 dtype = torch.bfloat16 if bf16 else torch.float16
 base_llm = AutoModelForCausalLM.from_pretrained(
 llm_dir,
 torch_dtype=dtype,
 device_map=None,
 )
 base_llm = base_llm.to(device)
 else:
 print(f"[Eval] WARN: llm_dir not found: {llm_dir}, using base LLM from config")
 print(f"[Eval] Initializing base LLM from config: {llm_name_or_path}")
 base_llm = init_llm(llm_name_or_path, tokenizer, bf16, device)

 print(f"[Eval] Building MolAwareCausalLM skeleton ...")
 model = init_model(cfg, tokenizer, base_llm, device)
 
 # model.llmcorrectLLM
 model.llm = base_llm

 # ------------ ckpt_dir/extras load GVP / MLP / Diffusion weight ------------
 extras_dir = os.path.join(ckpt_dir, "extras")
 if os.path.isdir(extras_dir):
 def _try_load_extra(attr_name: str, filename: str):
 path = os.path.join(extras_dir, filename)
 if os.path.exists(path) and hasattr(model, attr_name):
 try:
 print(f"[Eval] Loading {attr_name} from: {path}")
 state = torch.load(path, map_location=device)
 getattr(model, attr_name).load_state_dict(state, strict=False)
 except Exception as e:
 print(f"[Eval] WARN: failed to load {attr_name} from {path}: {e}")

 _try_load_extra("gvp_encoder", "gvp_encoder.pt")
 _try_load_extra("mol_adapter", "mol_adapter.pt")
 _try_load_extra("diffusion_adapter", "diffusion_adapter.pt")
 else:
 print(f"[Eval] WARN: extras_dir not found: {extras_dir}, skip loading extras")

 model.eval()
 return model, tokenizer


# ------------------------- evaluation ------------------------- #

def prepare_qm9_indices(dataset, max_samples: Optional[int]) -> List[int]:
 """ QM9 regression sample index max_samples """
 indices = []
 for i, ex in enumerate(dataset):
 if ex.get("dataset") != "QM9":
 continue
 if ex.get("task_type") != "regression":
 continue
 prop_name = ex.get("property_name", None)
 if prop_name not in QM9_TASKS:
 continue
 all_targets = ex.get("all_targets", None)
 if not isinstance(all_targets, dict):
 continue
 if prop_name not in all_targets:
 continue
 indices.append(i)
 if max_samples is not None:
 indices = indices[:max_samples]
 print(f"[Eval] Found {len(indices)} QM9 regression samples for evaluation")
 return indices


def run_one_inference(
 model: MolAwareCausalLM,
 tokenizer,
 device: str,
 user_text: str,
 realtime_mol: bool,
 max_new_tokens: int,
 temperature: float,
 top_p: float,
 top_k: int,
) -> Tuple[str, Optional[float]]:
 """
 user_text generatereturns (caption, parsed_value)
 """
 if not user_text:
 return "", None

 # tokenizerpad_token
 if tokenizer.pad_token is None:
 tokenizer.pad_token = tokenizer.eos_token
 tokenizer.pad_token_id = tokenizer.eos_token_id

 # usepromptbuildfunctionsupportsmodel
 prompt = build_prompt_from_tokenizer(user_text, tokenizer)
 inputs = tokenizer(
 prompt,
 return_tensors="pt",
 add_special_tokens=False,
 truncation=True,
 max_length=tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 4096,
 ).to(device)

 # validateinput_idswhethervalidrange
 vocab_size = len(tokenizer) if hasattr(tokenizer, '__len__') else tokenizer.vocab_size
 input_ids = inputs["input_ids"]
 if torch.any(input_ids >= vocab_size) or torch.any(input_ids < 0):
 invalid_indices = torch.where((input_ids >= vocab_size) | (input_ids < 0))
 print(f"[WARN] Invalid token IDs found: {input_ids[invalid_indices]}")
 print(f"[WARN] Vocab size: {vocab_size}, Input shape: {input_ids.shape}")
 # invalidtoken IDreplaceunk_token_ideos_token_id
 unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id
 input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
 inputs["input_ids"] = input_ids

 try:
 with torch.no_grad():
 gen_ids = model.generate(
 input_ids=inputs["input_ids"],
 attention_mask=inputs.get("attention_mask"),
 realtime_mol=realtime_mol,
 max_new_tokens=max_new_tokens,
 do_sample=(temperature > 0),
 temperature=temperature if temperature > 0 else 1.0,
 top_p=top_p,
 top_k=top_k,
 eos_token_id=tokenizer.eos_token_id,
 repetition_penalty=1.06,
 no_repeat_ngram_size=3,
 pad_token_id=tokenizer.pad_token_id,
 )

 gen_new = gen_ids[0, inputs["input_ids"].shape[1]:]
 caption = tokenizer.decode(gen_new, skip_special_tokens=True)
 val = extract_first_float(caption)
 return caption, val
 except RuntimeError as e:
 if "device-side assert" in str(e) or "CUDA error" in str(e):
 print(f"[ERROR] CUDA error during generation: {e}")
 print(f"[ERROR] Input text: {user_text[:200]}")
 # tensorCPUcheckCUDAerrorcontinueCUDA tensor
 try:
 input_ids_cpu = inputs['input_ids'].cpu()
 print(f"[ERROR] Input IDs shape: {input_ids_cpu.shape}")
 print(f"[ERROR] Input IDs max: {input_ids_cpu.max().item()}, min: {input_ids_cpu.min().item()}")
 print(f"[ERROR] Vocab size: {vocab_size}")
 # checkwhetherrangetoken
 invalid_mask = (input_ids_cpu >= vocab_size) | (input_ids_cpu < 0)
 if invalid_mask.any():
 invalid_ids = input_ids_cpu[invalid_mask].unique().tolist()
 print(f"[ERROR] Invalid token IDs found: {invalid_ids}")
 except Exception as debug_e:
 print(f"[ERROR] Failed to debug: {debug_e}")
 return "", None
 else:
 raise


def eval_qm9_compare(
 model: MolAwareCausalLM,
 tokenizer,
 dataset,
 indices: List[int],
 device: str,
 max_new_tokens: int = 64,
 temperature: float = 0.0,
 top_p: float = 1.0,
 top_k: int = 0,
) -> Dict[str, Any]:
 """
 samplerunset

 1) baseline: use ex['input']realtime_mol=False
 2) with_gnn: use <mol>{smiles}</mol> promptrealtime_mol=True

 setsuccessparsevaluesamplemetric
 """

 y_true: Dict[str, List[float]] = {t: [] for t in QM9_TASKS}
 y_pred_baseline: Dict[str, List[float]] = {t: [] for t in QM9_TASKS}
 y_pred_with_gnn: Dict[str, List[float]] = {t: [] for t in QM9_TASKS}
 
 # saveeachsample
 detailed_results: List[Dict[str, Any]] = []

 for idx in tqdm(indices, desc="Evaluating QM9 (baseline vs with_gnn)", ncols=100):
 ex = dataset[idx]
 prop_name = ex.get("property_name", None)
 if prop_name not in QM9_TASKS:
 continue

 all_targets = ex.get("all_targets", None)
 if not isinstance(all_targets, dict) or prop_name not in all_targets:
 continue
 gt_val = float(all_targets[prop_name])

 # ---------- baseline promptuseoriginal input ----------
 base_user_text = ex.get("input", "")
 # ---------- with_gnn promptuse <mol>{smiles}</mol> + value ----------
 smiles = ex.get("smiles", "")
 if smiles:
 prop_symbol = ex.get("property_symbol", prop_name)
 unit = ex.get("unit", "")
 suffix = f" in {unit}" if unit else ""
 gnn_user_text = (
 f"Given the molecule <mol>{smiles}</mol>, "
 f"what is its {prop_symbol}{suffix}? "
 f"Answer with only one numeric value."
 )
 else:
 # smiles input
 gnn_user_text = base_user_text

 # ---------- baseline inference ----------
 text_base, val_base = run_one_inference(
 model=model,
 tokenizer=tokenizer,
 device=device,
 user_text=base_user_text,
 realtime_mol=False,
 max_new_tokens=max_new_tokens,
 temperature=temperature,
 top_p=top_p,
 top_k=top_k,
 )

 # ---------- with_gnn inference ----------
 text_gnn, val_gnn = run_one_inference(
 model=model,
 tokenizer=tokenizer,
 device=device,
 user_text=gnn_user_text,
 realtime_mol=True,
 max_new_tokens=max_new_tokens,
 temperature=temperature,
 top_p=top_p,
 top_k=top_k,
 )

 # recordwhetherparsesuccess
 sample_detail = {
 "index": idx,
 "property_name": prop_name,
 "property_symbol": ex.get("property_symbol", prop_name),
 "unit": ex.get("unit", ""),
 "ground_truth": gt_val,
 "smiles": smiles,
 "input_text": base_user_text,
 "baseline": {
 "input": base_user_text,
 "output": text_base,
 "predicted_value": val_base,
 "error": abs(val_base - gt_val) if val_base is not None else None,
 "relative_error": abs(val_base - gt_val) / abs(gt_val) if val_base is not None and gt_val != 0 else None,
 },
 "with_gnn": {
 "input": gnn_user_text,
 "output": text_gnn,
 "predicted_value": val_gnn,
 "error": abs(val_gnn - gt_val) if val_gnn is not None else None,
 "relative_error": abs(val_gnn - gt_val) / abs(gt_val) if val_gnn is not None and gt_val != 0 else None,
 },
 "status": "used" if (val_base is not None and val_gnn is not None) else "skipped",
 }
 detailed_results.append(sample_detail)

 # parsefailskip
 if val_base is None or val_gnn is None:
 continue

 y_true[prop_name].append(gt_val)
 y_pred_baseline[prop_name].append(val_base)
 y_pred_with_gnn[prop_name].append(val_gnn)

 # ---------- statisticsmetric ----------
 per_task_metrics: Dict[str, Dict[str, Any]] = {}
 maes_baseline, maes_gnn = [], []
 rmses_baseline, rmses_gnn = [], []

 print("\n========== Per-Task Metrics ==========")
 for t in QM9_TASKS:
 yt = y_true[t]
 yb = y_pred_baseline[t]
 yg = y_pred_with_gnn[t]
 if len(yt) == 0:
 print(f"[Eval] WARN: No valid samples for task '{t}'")
 continue

 m_base = compute_regression_metrics(yt, yb)
 m_gnn = compute_regression_metrics(yt, yg)
 per_task_metrics[t] = {
 "baseline": m_base,
 "with_gnn": m_gnn,
 }

 maes_baseline.append(m_base["mae"])
 maes_gnn.append(m_gnn["mae"])
 rmses_baseline.append(m_base["rmse"])
 rmses_gnn.append(m_gnn["rmse"])

 print(
 f"Task {t:5s} | n={m_base['n']:4d} | "
 f"Baseline MAE={m_base['mae']:.6f}, RMSE={m_base['rmse']:.6f} || "
 f"WithGNN MAE={m_gnn['mae']:.6f}, RMSE={m_gnn['rmse']:.6f}"
 )

 if maes_baseline and maes_gnn:
 avg_base_mae = float(np.mean(maes_baseline))
 avg_gnn_mae = float(np.mean(maes_gnn))
 avg_base_rmse = float(np.mean(rmses_baseline))
 avg_gnn_rmse = float(np.mean(rmses_gnn))
 else:
 avg_base_mae = avg_gnn_mae = float("nan")
 avg_base_rmse = avg_gnn_rmse = float("nan")

 summary = {
 "per_task": per_task_metrics,
 "avg_baseline_mae": avg_base_mae,
 "avg_baseline_rmse": avg_base_rmse,
 "avg_with_gnn_mae": avg_gnn_mae,
 "avg_with_gnn_rmse": avg_gnn_rmse,
 "detailed_results": detailed_results, # result
 }

 print("\n========== QM9 Multi-task Summary ==========")
 print(f"Average Baseline | MAE={avg_base_mae:.6f} | RMSE={avg_base_rmse:.6f}")
 print(f"Average With GNN | MAE={avg_gnn_mae:.6f} | RMSE={avg_gnn_rmse:.6f}")
 print("============================================\n")

 return summary


# ------------------------- CLI ------------------------- #

def parse_args():
 parser = argparse.ArgumentParser(description="QM9 Multi-task Comparison (Baseline vs MolAware+GNN)")
 parser.add_argument("--cfg", type=str, required=True, help="YAML config file (e.g. epoch2_config_modified.yaml)")
 parser.add_argument("--ckpt_dir", type=str, required=True,
 help="Checkpoint dir for Stage2 (containing llm/ and extras/), e.g. .../epoch2/checkpoint-4000")
 parser.add_argument("--test_file", type=str, required=True, help="QM9 test jsonl file (contains dataset/task_type/all_targets )")
 parser.add_argument("--max_samples", type=int, default=None, help="maxevaluationsample QM9 regression sample")
 parser.add_argument("--out_dir", type=str, required=True, help="outputmetric JSON directory")
 parser.add_argument("--max_new_tokens", type=int, default=64)
 parser.add_argument("--temperature", type=float, default=0.0)
 parser.add_argument("--top_p", type=float, default=1.0)
 parser.add_argument("--top_k", type=int, default=0)
 parser.add_argument("--seed", type=int, default=42)
 parser.add_argument("--device", type=str, default=None, help="cuda:0 / cuda / cpu (default: auto)")
 return parser.parse_args()


def main():
 args = parse_args()
 set_seed(args.seed)

 # ---- load config ----
 with open(args.cfg, "r") as f:
 cfg = yaml.safe_load(f)

 ckpt_dir = args.ckpt_dir
 if not os.path.isdir(ckpt_dir):
 raise FileNotFoundError(f"ckpt_dir not found: {ckpt_dir}")

 # ---- device ----
 if args.device is not None:
 device = args.device
 else:
 device = "cuda" if torch.cuda.is_available() else "cpu"

 print(f"[Eval] Using device: {device}")
 if device.startswith("cuda"):
 # 0multi-GPU
 torch.cuda.set_device(0)

 # ---- loadmodel ----
 model, tokenizer = load_molaware_from_ckpt(cfg, ckpt_dir, device)

 # ---- loadtest ----
 if not os.path.exists(args.test_file):
 raise FileNotFoundError(f"Test file not found: {args.test_file}")
 print(f"[Eval] Loading test set from: {args.test_file}")
 dataset = load_dataset("json", data_files=args.test_file, split="train")

 # ---- QM9 sample index ----
 indices = prepare_qm9_indices(dataset, max_samples=args.max_samples)

 # ---- comparisonevaluation ----
 summary = eval_qm9_compare(
 model=model,
 tokenizer=tokenizer,
 dataset=dataset,
 indices=indices,
 device=device,
 max_new_tokens=args.max_new_tokens,
 temperature=args.temperature,
 top_p=args.top_p,
 top_k=args.top_k,
 )

 # ---- saveresult ----
 os.makedirs(args.out_dir, exist_ok=True)
 baseline_path = os.path.join(args.out_dir, "qm9_baseline.json")
 gnn_path = os.path.join(args.out_dir, "qm9_with_gnn.json")

 # summary file
 per_task = summary["per_task"]
 baseline_only = {
 "per_task": {k: v["baseline"] for k, v in per_task.items()},
 "avg_mae": summary["avg_baseline_mae"],
 "avg_rmse": summary["avg_baseline_rmse"],
 }
 gnn_only = {
 "per_task": {k: v["with_gnn"] for k, v in per_task.items()},
 "avg_mae": summary["avg_with_gnn_mae"],
 "avg_rmse": summary["avg_with_gnn_rmse"],
 }

 with open(baseline_path, "w", encoding="utf-8") as f:
 json.dump(baseline_only, f, ensure_ascii=False, indent=2)
 with open(gnn_path, "w", encoding="utf-8") as f:
 json.dump(gnn_only, f, ensure_ascii=False, indent=2)
 print(f"[Eval] Baseline metrics saved to: {baseline_path}")
 print(f"[Eval] With-GNN metrics saved to: {gnn_path}")
 
 # saveresulteachsampleprediction
 detailed_path = os.path.join(args.out_dir, "qm9_detailed_results.jsonl")
 detailed_results = summary.get("detailed_results", [])
 if detailed_results:
 with open(detailed_path, "w", encoding="utf-8") as f:
 for result in detailed_results:
 f.write(json.dumps(result, ensure_ascii=False) + "\n")
 print(f"[Eval] Detailed results saved to: {detailed_path} ({len(detailed_results)} samples)")


if __name__ == "__main__":
 main()
