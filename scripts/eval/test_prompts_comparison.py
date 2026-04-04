#!/usr/bin/env python3
"""
testscriptdifferentmodelinferenceresult
- modelllasmol, intern-s1
- inferencesft_tester, inferencetransformers
- testalljsonlfilefirstprompt

usemethod
 python test_prompts_comparison.py --model llasmol --type sft_tester
 python test_prompts_comparison.py --model llasmol --type transformers
 python test_prompts_comparison.py --model intern-s1 --type sft_tester
 python test_prompts_comparison.py --model intern-s1 --type transformers
"""

import json
import os
import sys
import argparse
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# path
BASE_DIR = Path("${SCICORE_ROOT:-/path/to/scicore-mol}")
sys.path.insert(0, str(BASE_DIR))

from sft_tester import MolAwareGenerator2

# config
TEST_DIR = BASE_DIR / "1223results_baseline/LlaSMol-Mistral-7B-merged_fewshot"
MODEL_DIR = Path("${DATA_DIR:-/path/to/data}/base_model")

# model pathconfig
MODELS = {
 "llasmol": MODEL_DIR / "LlaSMol-Mistral-7B-merged",
 "intern-s1": MODEL_DIR / "Intern-S1-mini",
}

# Intern-S1 needs base_llm_path
BASE_LLM_PATHS = {
 "llasmol": None,
 "intern-s1": MODEL_DIR / "qwen3_8b",
}

TOKEN_CLS_PATH = "${CHECKPOINT_DIR:-/path/to/checkpoints}/llama_mlp_token_classifier.pt"
OUTPUT_FILE = BASE_DIR / "1223results_baseline/prompt_comparison_results.jsonl"

# processfunction
import re

YESNO_RE = re.compile(r"\b(Yes|No)\b", re.IGNORECASE)
FLOAT_RE = re.compile(r"[-+]?\d+(\.\d+)?")

# jsonl filekey
EXCLUDE_SUBSTR = ["metrics", "eval_summary", "evaluation", "result", "score"]


def safe_apply_chat_template(tokenizer, messages, extra_kwargs=None):
 """call apply_chat_templatesupportsparameter"""
 extra_kwargs = extra_kwargs or {}
 if not hasattr(tokenizer, "apply_chat_template") or getattr(tokenizer, "chat_template", None) is None:
 return None
 try:
 sig = inspect.signature(tokenizer.apply_chat_template)
 supported = set(sig.parameters.keys())
 kwargs = {"tokenize": False, "add_generation_prompt": True}
 for k, v in extra_kwargs.items():
 if k in supported:
 kwargs[k] = v
 return tokenizer.apply_chat_template(messages, **kwargs)
 except Exception:
 return None


def format_chat_prompt(tokenizer, prompt, model_name, system_msg="You are a careful chemist. Follow the requested output format exactly."):
 """format chat promptinferencepathusesameformat"""
 messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}]
 extra = {}
 if model_name == "intern-s1":
 extra["enable_thinking"] = False
 text = safe_apply_chat_template(tokenizer, messages, extra_kwargs=extra)
 if text is not None:
 return text
 # fallback
 return f"System: {system_msg}\n\nUser: {prompt}\n\nAssistant: "


def infer_input_device(model):
 """inferinputshoulddeviceprocessmulti-GPU"""
 if hasattr(model, "device") and str(model.device) not in ("meta", "cpu"):
 return model.device
 if hasattr(model, "hf_device_map"):
 # embedding key
 for k, v in model.hf_device_map.items():
 if any(x in k for x in ["embed", "wte", "word_embeddings", "tok_embeddings"]):
 return v
 # first cuda
 for v in model.hf_device_map.values():
 if isinstance(v, str) and v.startswith("cuda"):
 return v
 return "cuda:0" if torch.cuda.is_available() else "cpu"


def strip_thinking(t: str) -> str:
 """extract thinking label"""
 if not t:
 return t
 # process think label
 if "<think>" in t and "</think>" in t:
 return t.split("</think>", 1)[1].strip()
 # ""
 THINK_SPLITS = ["</think>", "</think>", "</thinking>", "<|im_end|>", "<|endoftext|>"]
 for sp in THINK_SPLITS:
 if sp in t:
 tail = t.split(sp)[-1].strip()
 if tail:
 return tail
 return t


def clean_output(task: str, text: str) -> str:
 """processoutputformatextract"""
 if text is None or not text:
 return ""
 
 # 1) special token / 
 t = text.replace("<|im_end|>", "").replace("</s>", "").strip()
 
 # 2) process thinkingextract
 t = strip_thinking(t)
 
 # 3) extractlabelif
 # <SMILES> ... </SMILES> / <SOL> ... </SOL> / <SOLUTION> ... </SOLUTION>
 for tag in ["SMILES", "SOL", "SOLUTION", "MOLFORMULA"]:
 m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", t, flags=re.S)
 if m:
 t = m.group(1).strip()
 break
 
 # 4) task
 t = t.splitlines()[0].strip()
 
 # 5) task-specific 
 if task.startswith("property_prediction-") and task not in ["property_prediction-esol", "property_prediction-lipo"]:
 m = YESNO_RE.search(t)
 if m:
 return "Yes" if m.group(1).lower() == "yes" else "No"
 return t # returns
 
 if task in ["property_prediction-esol", "property_prediction-lipo"]:
 m = FLOAT_RE.search(t)
 return m.group(0) if m else t
 
 # smiles classtaskempty
 if task in ["forward_synthesis", "retrosynthesis", "molecule_generation"]:
 return t.replace(" ", "")
 
 return t


def gen_config_for_task(task: str) -> Dict[str, Any]:
 """according totasktypereturnsgenerateconfig"""
 if task.startswith("property_prediction-") and task not in ["property_prediction-esol", "property_prediction-lipo"]:
 # Yes/No classtaskgreedy + max_new_tokens
 return {
 "max_new_tokens": 8,
 "do_sample": False,
 "temperature": 0.0,
 "top_p": 1.0,
 "repetition_penalty": 1.06,
 "no_repeat_ngram_size": 3,
 }
 if task in ["property_prediction-esol", "property_prediction-lipo"]:
 # valueregressiontask
 return {
 "max_new_tokens": 32,
 "do_sample": False,
 "temperature": 0.0,
 "top_p": 1.0,
 "repetition_penalty": 1.06,
 "no_repeat_ngram_size": 3,
 }
 if task in ["forward_synthesis", "retrosynthesis", "molecule_generation"]:
 # SMILES generatetask
 return {
 "max_new_tokens": 256,
 "do_sample": False,
 "temperature": 0.0,
 "top_p": 1.0,
 "repetition_penalty": 1.06,
 "no_repeat_ngram_size": 3,
 }
 # taskname_conversion, molecule_captioning 
 return {
 "max_new_tokens": 256,
 "do_sample": True,
 "temperature": 0.2,
 "top_p": 0.9,
 "repetition_penalty": 1.06,
 "no_repeat_ngram_size": 3,
 }


def load_first_prompt_from_jsonl(jsonl_path: Path) -> Dict[str, Any]:
 """jsonlfileloadfirstprompt"""
 with open(jsonl_path, 'r', encoding='utf-8') as f:
 first_line = f.readline().strip()
 if first_line:
 return json.loads(first_line)
 return None


def find_all_jsonl_files(test_dir: Path) -> List[Path]:
 """alljsonlfileevaluationresultfile"""
 jsonl_files = []
 
 for file in test_dir.glob("*.jsonl"):
 # containskey
 if any(s in file.name.lower() for s in EXCLUDE_SUBSTR):
 continue
 jsonl_files.append(file)
 
 return sorted(jsonl_files)


def load_sft_tester_generator(
 model_name: str,
 model_path: Path,
 base_llm_path: Path = None
) -> MolAwareGenerator2:
 """load sft_tester generatorload"""
 generator = MolAwareGenerator2()
 
 cfg = {
 "ckpt_dir": str(model_path),
 "device": "cuda:0" if torch.cuda.is_available() else "cpu",
 "dtype": "bf16",
 "debug": False,
 "enable_thinking": False,
 "realtime_mol": False,
 }
 
 if base_llm_path:
 cfg["base_llm_path"] = str(base_llm_path)
 
 if model_name == "llasmol":
 cfg["token_classifier_path"] = TOKEN_CLS_PATH
 
 generator.load(cfg)
 return generator


def inference_with_sft_tester(
 generator: MolAwareGenerator2,
 prompt: str,
 task_name: str,
 model_name: str
) -> str:
 """useload sft_tester generator inference"""
 # according totasktypechoosegenerateconfig
 gen_config = gen_config_for_task(task_name)
 
 # SMILES taskproperty_prediction, name_conversion-s2i
 # llasmol sft_tester filtermolecule transformers
 # ifreturnsemptyprocess
 
 # Intern-S1 needs thinking
 kwargs = {
 "add_dialog_wrapper": True,
 "realtime_mol": False,
 **gen_config
 }
 
 # if generator supports enable_thinking parameter
 # NOTE sft_tester implement
 try:
 result = generator.generate(prompt, **kwargs)
 except TypeError:
 # ifsupportsparameteruseparameter
 result = generator.generate(
 prompt,
 add_dialog_wrapper=True,
 realtime_mol=False,
 max_new_tokens=gen_config.get("max_new_tokens", 256),
 temperature=gen_config.get("temperature", 0.2),
 top_p=gen_config.get("top_p", 0.9),
 repetition_penalty=gen_config.get("repetition_penalty", 1.06),
 no_repeat_ngram_size=gen_config.get("no_repeat_ngram_size", 3),
 )
 
 return result


def load_transformers_model(
 model_name: str,
 model_path: Path,
 base_llm_path: Path = None
) -> tuple:
 """load transformers model tokenizerload"""
 # load tokenizer
 if base_llm_path and model_name == "intern-s1":
 tokenizer_path = base_llm_path
 else:
 tokenizer_path = model_path
 
 try:
 tokenizer = AutoTokenizer.from_pretrained(
 str(tokenizer_path),
 trust_remote_code=True
 )
 except Exception as e:
 raise RuntimeError(f"load tokenizer fail: {e}")
 
 # loadmodel
 try:
 model = AutoModelForCausalLM.from_pretrained(
 str(model_path),
 torch_dtype=torch.bfloat16, # use torch_dtype API
 device_map="auto",
 trust_remote_code=True
 )
 model.eval()
 except Exception as e:
 raise RuntimeError(f"loadmodelfail: {e}")
 
 return tokenizer, model


def inference_with_transformers(
 model_name: str,
 tokenizer: AutoTokenizer,
 model: AutoModelForCausalLM,
 prompt: str,
 task_name: str
) -> str:
 """useload transformers modelinference"""
 # according totasktypechoosegenerateconfig
 gen_config = gen_config_for_task(task_name)
 
 # format promptuse format_chat_prompt sft_tester consistent
 system_msg = "You are a careful chemist. Follow the requested output format exactly."
 formatted_prompt = format_chat_prompt(tokenizer, prompt, model_name, system_msg)
 
 # Tokenize
 try:
 inputs = tokenizer(formatted_prompt, return_tensors="pt")
 input_ids_len = inputs["input_ids"].shape[1] # saveoriginallengthfordecode
 
 # moveinputmodeldeviceusedeviceinfer
 dev = infer_input_device(model)
 inputs = {k: v.to(dev) for k, v in inputs.items()}
 except Exception as e:
 raise RuntimeError(f"Tokenize fail: {e}")
 
 # Generate
 try:
 # set eos_token_id pad_token_id
 generate_kwargs = {
 **inputs,
 **gen_config,
 }
 
 # set eos_token_id
 if "eos_token_id" not in generate_kwargs:
 generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
 if "pad_token_id" not in generate_kwargs:
 generate_kwargs["pad_token_id"] = tokenizer.eos_token_id
 
 with torch.no_grad():
 outputs = model.generate(**generate_kwargs)
 except Exception as e:
 raise RuntimeError(f"generatefail: {e}")
 
 # Decode
 try:
 # intern-s1 skip special tokenskey
 # via clean_output process
 skip_special = model_name != "intern-s1"
 generated_text = tokenizer.decode(
 outputs[0][input_ids_len:],
 skip_special_tokens=skip_special
 )
 except Exception as e:
 raise RuntimeError(f"decodefail: {e}")
 
 return generated_text


def collect_all_prompts(jsonl_files: List[Path]) -> List[Dict[str, Any]]:
 """alltask prompt"""
 tasks = []
 print(f"📋 alltask prompt...")
 
 for jsonl_file in jsonl_files:
 task_name = jsonl_file.stem
 print(f" 📝 {task_name}...")
 
 # loadfirst prompt
 data = load_first_prompt_from_jsonl(jsonl_file)
 if not data:
 print(f" ⚠️ skipfileempty")
 continue
 
 prompt = data.get("prompt", "")
 gold = data.get("gold", "")
 input_text = data.get("input", "")
 
 if not prompt:
 print(f" ⚠️ skip prompt")
 continue
 
 tasks.append({
 "task": task_name,
 "prompt": prompt,
 "input": input_text,
 "gold": gold,
 })
 print(f" ✓ Prompt length: {len(prompt)} ")
 
 print(f"\n✅ {len(tasks)} task\n")
 return tasks


def parse_args():
 """parseparameter"""
 parser = argparse.ArgumentParser(
 description="testscriptdifferentmodelinferenceresult",
 formatter_class=argparse.RawDescriptionHelpFormatter,
 epilog="""
exampleneedsrundifferent setting
 python test_prompts_comparison.py --model llasmol --type sft_tester
 python test_prompts_comparison.py --model llasmol --type transformers
 python test_prompts_comparison.py --model intern-s1 --type sft_tester
 python test_prompts_comparison.py --model intern-s1 --type transformers
 """
 )
 parser.add_argument(
 "--model",
 type=str,
 required=True,
 choices=["llasmol", "intern-s1"],
 help="usemodel"
 )
 parser.add_argument(
 "--type",
 type=str,
 required=True,
 choices=["sft_tester", "transformers"],
 help="inference"
 )
 parser.add_argument(
 "--output",
 type=str,
 default=None,
 help="outputfilepathdefaultaccording to model type generate"
 )
 return parser.parse_args()


def main():
 """function"""
 args = parse_args()
 model_name = args.model
 inference_type = args.type
 
 print(f"🚀 runset: model={model_name}, type={inference_type}\n")
 
 # checkmodel path
 if model_name not in MODELS:
 print(f"❌ model: {model_name}")
 return
 
 model_path = MODELS[model_name]
 if not model_path.exists():
 print(f"❌ model path: {model_path}")
 return
 
 base_llm_path = BASE_LLM_PATHS.get(model_name)
 if base_llm_path and not base_llm_path.exists():
 print(f"❌ Base LLM path: {base_llm_path}")
 return
 
 # step1: testfilealltask prompt
 print(f"🔍 testfile...")
 jsonl_files = find_all_jsonl_files(TEST_DIR)
 print(f" {len(jsonl_files)} jsonl file\n")
 
 tasks = collect_all_prompts(jsonl_files)
 
 if not tasks:
 print("❌ validtask")
 return
 
 # step2: loadmodelinference
 print(f"📦 loadmodel: {model_name} ({inference_type})...")
 
 loaded_model = None
 if inference_type == "sft_tester":
 try:
 sft_generator = load_sft_tester_generator(model_name, model_path, base_llm_path)
 loaded_model = sft_generator
 print(f"✅ sft_tester generator loadcomplete")
 except Exception as e:
 print(f"❌ sft_tester generator loadfail: {e}")
 return
 else: # transformers
 try:
 tokenizer, model = load_transformers_model(model_name, model_path, base_llm_path)
 loaded_model = (tokenizer, model)
 print(f"✅ transformers modelloadcomplete")
 except Exception as e:
 print(f"❌ transformers modelloadfail: {e}")
 return
 
 print(f"\n✅ modelloadcompletestartprocess {len(tasks)} task...\n")
 
 # step3: processalltask
 results = []
 
 for task_idx, task_data in enumerate(tasks, 1):
 task_name = task_data["task"]
 prompt = task_data["prompt"]
 gold = task_data["gold"]
 input_text = task_data["input"]
 
 print(f"[{task_idx}/{len(tasks)}] 📝 processtask: {task_name}")
 print(f" Prompt length: {len(prompt)} ")
 
 try:
 if inference_type == "sft_tester":
 print(f" → sft_tester inference...")
 result_raw = inference_with_sft_tester(loaded_model, prompt, task_name, model_name)
 else: # transformers
 print(f" → transformers inference...")
 tokenizer, model = loaded_model
 result_raw = inference_with_transformers(model_name, tokenizer, model, prompt, task_name)
 
 # processoutput
 result_clean = clean_output(task_name, result_raw)
 
 results.append({
 "task": task_name,
 "model": model_name,
 "type": inference_type,
 "prompt": prompt,
 "input": input_text,
 "gold": gold,
 "prediction_raw": result_raw, # originaloutputfor debug
 "prediction": result_clean, # outputforevaluation
 })
 
 # ifemptyoriginalemptywarning
 if not result_clean and result_raw:
 print(f" ⚠️ warningoutputemptyoriginaloutputlength: {len(result_raw)}")
 
 print(f" ✓ complete (original: {len(result_raw)} , : {len(result_clean)} )")
 except Exception as e:
 print(f" ✗ fail: {e}")
 import traceback
 traceback.print_exc()
 results.append({
 "task": task_name,
 "model": model_name,
 "type": inference_type,
 "prompt": prompt,
 "input": input_text,
 "gold": gold,
 "prediction_raw": f"ERROR: {str(e)}",
 "prediction": f"ERROR: {str(e)}",
 })
 
 # outputfilepath
 if args.output:
 output_file = Path(args.output)
 else:
 # according to model type generatefile
 output_file = OUTPUT_FILE.parent / f"prompt_comparison_{model_name}_{inference_type}.jsonl"
 
 # saveresult
 output_file.parent.mkdir(parents=True, exist_ok=True)
 with open(output_file, 'w', encoding='utf-8') as f:
 for result in results:
 f.write(json.dumps(result, ensure_ascii=False) + '\n')
 
 print(f"\n✅ completeresultsave: {output_file}")
 print(f" {len(results)} result")
 print(f" set: model={model_name}, type={inference_type}")


if __name__ == "__main__":
 main()

