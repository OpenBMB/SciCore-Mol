#!/usr/bin/env python3
"""
stagegenerate LLM trainingdata
Pipeline: query -> LLM -> Layer2 -> generatetrainingdata

generatedataformat
{
 "input": "original query",
 "intermediate": " LLM outputcontainsreaction SMILES",
 "layer2_info": {
 "yield_bin": 5,
 "yield_reg": 0.75,
 "embedding": [...]
 },
 "output": " LLM outputuse layer2 "
}
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch

# directorypath
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sft_tester import MolAwareGenerator2


def load_queries(input_path: str) -> List[Dict[str, Any]]:
 """loaddata"""
 queries = []
 with open(input_path, 'r', encoding='utf-8') as f:
 for line in f:
 line = line.strip()
 if not line:
 continue
 try:
 data = json.loads(line)
 queries.append(data)
 except json.JSONDecodeError:
 # if JSON
 queries.append({"input": line})
 return queries


def load_chembench_data(task: str = "product", split: str = "train") -> List[Dict[str, Any]]:
 """ ChemBench loaddata"""
 try:
 from datasets import load_dataset
 except ImportError:
 raise RuntimeError("Please install datasets: pip install datasets")
 
 REPO_ID = "AI4Chem/ChemBench4K"
 BENCH_FILES = {
 "product": {
 "dev": "dev/Product_Prediction_benchmark.json",
 "test": "test/Product_Prediction_benchmark.json",
 },
 "retro": {
 "dev": "dev/Retrosynthesis_benchmark.json",
 "test": "test/Retrosynthesis_benchmark.json",
 },
 "yield": {
 "dev": "dev/Yield_Prediction_benchmark.json",
 "test": "test/Yield_Prediction_benchmark.json",
 },
 }
 
 if task not in BENCH_FILES:
 raise ValueError(f"Unsupported task: {task}")
 
 # ChemBench train splituse dev trainingdata
 if split == "train":
 print("[INFO] ChemBench train splituse dev trainingdata")
 split = "dev"
 
 if split not in BENCH_FILES[task]:
 raise ValueError(f"Unsupported split: {split}")
 
 relpath = BENCH_FILES[task][split]
 url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{relpath}"
 
 print(f"[INFO] Loading ChemBench data: {task}/{split}")
 print(f"[INFO] URL: {url}")
 
 ds = load_dataset("json", data_files={split: url}, split=split)
 
 # convertformat
 queries = []
 for sample in ds:
 question = sample.get("question", "")
 queries.append({"input": question})
 
 print(f"[INFO] Loaded {len(queries)} samples from ChemBench")
 return queries


def save_results(results: List[Dict[str, Any]], output_path: str):
 """saveresult JSONL"""
 with open(output_path, 'w', encoding='utf-8') as f:
 for item in results:
 f.write(json.dumps(item, ensure_ascii=False) + '\n')


def generate_training_data(
 generator: MolAwareGenerator2,
 queries: List[Dict[str, Any]],
 output_path: str,
 task_type: Optional[str] = None,
):
 """
 generatetrainingdata
 
 Args:
 generator: MolAwareGenerator2 
 queries: list
 output_path: outputfilepath
 task_type: tasktype "reaction_prediction"
 """
 results = []
 
 for i, query_data in enumerate(tqdm(queries, desc="generatetrainingdata")):
 query = query_data.get("input", query_data.get("query", ""))
 if not query:
 continue
 
 try:
 # use generate_with_layer2 generate pipeline resultgetresult
 result = generator.generate_with_layer2(
 prompt=query,
 add_dialog_wrapper=True,
 max_new_tokens=512,
 do_sample=True,
 temperature=0.7,
 task_type=task_type,
 return_intermediate=True, # returnsresult
 )
 
 # result dictcontains first_response, layer2_info, final_response
 if isinstance(result, dict):
 first_response = result.get("first_response", "")
 layer2_info = result.get("layer2_info", {})
 final_response = result.get("final_response", "")
 
 # parseoutput JSONcontainsmolecule
 molecules_info = None
 try:
 import json
 import re
 # first_response extract JSON
 json_match = re.search(r'\{.*\}', first_response, re.DOTALL)
 if json_match:
 json_str = json_match.group(0)
 parsed = json.loads(json_str)
 if "molecules" in parsed:
 molecules_info = parsed["molecules"]
 except:
 pass
 
 # buildtrainingdata
 training_item = {
 "input": query,
 "intermediate": first_response, # JSON output
 "molecules_info": molecules_info, # parsemoleculecontains
 "layer2_info": {
 "yield_bin": layer2_info.get("yield_bin") if layer2_info else None,
 "yield_reg": layer2_info.get("yield_reg") if layer2_info else None,
 # embedding tensorsave embedding trainingdynamicgenerate
 "embedding_shape": list(layer2_info.get("embedding", torch.tensor([])).shape) if layer2_info and layer2_info.get("embedding") is not None else None,
 },
 "output": final_response, # LLM output
 }
 else:
 # interfaceifreturnsstring
 training_item = {
 "input": query,
 "intermediate": "",
 "layer2_info": {},
 "output": result,
 }
 
 results.append(training_item)
 
 except Exception as e:
 print(f"❌ process {i} : {e}")
 import traceback
 traceback.print_exc()
 continue
 
 # saveresult
 save_results(results, output_path)
 print(f"✅ save {len(results)} trainingdata {output_path}")


def main():
 # defaultpath
 DEFAULT_INPUT = None # defaultuse ChemBench
 DEFAULT_OUTPUT = "${SCICORE_ROOT:-/path/to/scicore-mol}/scripts/layer2_llm/data/training_data.jsonl"
 DEFAULT_CONFIG = "${SCICORE_ROOT:-/path/to/scicore-mol}/configs/qwen3_sft_epoch2_2.yaml"
 
 parser = argparse.ArgumentParser(description="generate Layer2-LLM trainingdata")
 parser.add_argument("--input", type=str, default=DEFAULT_INPUT, 
 help="inputfileJSONLifuse ChemBench data")
 parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, 
 help=f"outputtrainingdatafileJSONLdefault: {DEFAULT_OUTPUT}")
 parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, 
 help=f"modelconfigfilepathdefault: {DEFAULT_CONFIG}")
 parser.add_argument("--task_type", type=str, default="reaction_prediction", help="tasktypedefault: reaction_prediction")
 parser.add_argument("--device", type=str, default="cuda:0", help="devicedefault: cuda:0")
 
 # ChemBench parameter
 parser.add_argument("--use_chembench", action="store_true", help="use ChemBench dataif --input default")
 parser.add_argument("--chembench_task", type=str, default="product", choices=["product", "retro", "yield"], 
 help="ChemBench tasktypedefault: product")
 parser.add_argument("--chembench_split", type=str, default="train", choices=["train", "dev", "test"],
 help="ChemBench datadefault: train")
 
 args = parser.parse_args()
 
 # loadconfig
 if args.config.endswith('.yaml') or args.config.endswith('.yml'):
 import yaml
 with open(args.config, 'r', encoding='utf-8') as f:
 train_cfg = yaml.safe_load(f)
 else:
 # JSON
 with open(args.config, 'r', encoding='utf-8') as f:
 train_cfg = json.load(f)
 
 # trainingconfigconvertgenerateconfigformat
 # checkdevicewhether
 device = args.device
 if device.startswith("cuda:"):
 import torch
 if not torch.cuda.is_available():
 print(f"⚠️ warning: CUDA deviceset {device}")
 print(" use CPU check CUDA_VISIBLE_DEVICES environmentvariable")
 # if CUDA CPU
 device = "cpu"
 else:
 # check GPU whetherrange
 gpu_id = int(device.split(":")[-1])
 visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
 if visible_gpus:
 visible_list = [int(x) for x in visible_gpus.split(",") if x.strip().isdigit()]
 if visible_list:
 # CUDA_VISIBLE_DEVICES mapping GPU ID
 # ifset CUDA_VISIBLE_DEVICES=0,1,2,3 cuda:0 first GPU
 # if cuda:0shoulduse cuda:0alreadymapping
 # if ID listrangeusefirst GPU
 if gpu_id >= len(visible_list):
 device = "cuda:0" # usefirst GPUmapping cuda:0
 print(f"⚠️ warning: GPU {gpu_id} rangeuse cuda:0mapping GPU {visible_list[0]}")
 else:
 # ifset CUDA_VISIBLE_DEVICESusedevice
 pass
 
 cfg = {
 "ckpt_dir": train_cfg.get("paths", {}).get("checkpoint_dir") or train_cfg.get("paths", {}).get("llm_name_or_path"),
 "device": device,
 "dtype": "bf16", # defaultuse bf16
 "debug": False,
 }
 
 # token_classifier_pathif
 token_classifier_path = train_cfg.get("paths", {}).get("mlp_token_classifier_path")
 if token_classifier_path:
 cfg["token_classifier_path"] = token_classifier_path
 
 # default Layer2 Layer2 trainingdatageneratescript
 import yaml
 script_dir = Path(__file__).parent.resolve()
 project_root = script_dir.parent.parent
 layer2_config_path = project_root / "modules" / "layer2_component" / "layer2_config.yaml"
 if layer2_config_path.exists():
 with open(layer2_config_path, 'r', encoding='utf-8') as f:
 layer2_config = yaml.safe_load(f)
 # configfilepath Layer2Inferer load
 cfg["layer2"] = {
 "config_path": str(layer2_config_path),
 **layer2_config, # containsconfiguse
 }
 else:
 print(f"[WARNING] Layer2 config not found at {layer2_config_path}, using defaults")
 cfg["layer2"] = {
 "config_path": None,
 "checkpoint_path": "${SCICORE_ROOT:-/path/to/scicore-mol}/Layer2/ckpt/0115/layer2_pretrain.pt",
 "gvp_root": "${DATA_DIR:-/path/to/data}/MSMLM",
 "gvp_ckpt_path": "${CHECKPOINT_DIR:-/path/to/checkpoints}/gvp_weights_best.pt",
 }
 
 # train configset use_layer2
 if "train" not in cfg:
 cfg["train"] = {}
 cfg["train"]["use_layer2"] = True
 print(f"[INFO] Layer2 enabled in config (train.use_layer2=True)")
 
 # checkconfig
 if not cfg.get("ckpt_dir"):
 raise ValueError("configfile checkpoint_dir llm_name_or_path")
 
 print(f"📦 use checkpoint: {cfg['ckpt_dir']}")
 
 # initializegenerate
 print("📦 initializemodel...")
 generator = MolAwareGenerator2()
 generator.load(cfg)
 
 # load
 if args.input:
 print(f"📂 fileload: {args.input}")
 queries = load_queries(args.input)
 print(f" {len(queries)} ")
 elif args.use_chembench or not args.input:
 # ifinputfiledefaultuse ChemBench
 print(f"📂 ChemBench loaddata")
 queries = load_chembench_data(task=args.chembench_task, split=args.chembench_split)
 print(f" {len(queries)} ")
 else:
 print("❌ error: must --input use --use_chembench")
 return
 
 # generatetrainingdata
 print("🔄 startgeneratetrainingdata...")
 generate_training_data(
 generator=generator,
 queries=queries,
 output_path=args.output,
 task_type=args.task_type,
 )
 
 print("✅ complete")


if __name__ == "__main__":
 main()
