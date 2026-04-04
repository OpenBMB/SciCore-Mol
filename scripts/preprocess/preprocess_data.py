"""
dataprocessscript
originaldataprocesstrainingformatprocessepochdata
"""
import os
import json
import yaml
import random
from pathlib import Path
from datasets import load_dataset
from typing import Dict, List, Any, Optional


def safe_to_str(x):
 """convertstring"""
 if x is None:
 return ""
 if isinstance(x, (list, tuple)):
 return "\n".join(safe_to_str(xx) for xx in x)
 if isinstance(x, dict):
 return json.dumps(x, ensure_ascii=False)
 return str(x)


def preprocess_qa_data(
 qm9_file: str,
 bbbp_file: str,
 output_file: str,
 use_offline_spans: bool = False,
 qm9_max_samples: Optional[int] = None,
 seed: int = 42,
):
 """
 processfirstepochQAdataQM9 + BBBP
 
 Args:
 qm9_file: QM9datafilepath
 bbbp_file: BBBPdatafilepath
 output_file: outputfilepath
 use_offline_spans: whetheruseneedsmodelprocess
 qm9_max_samples: QM9datamaxsamplecountNoneuse
 seed: random
 """
 data_files = []
 if os.path.exists(qm9_file):
 data_files.append(qm9_file)
 if os.path.exists(bbbp_file):
 data_files.append(bbbp_file)
 
 if not data_files:
 raise FileNotFoundError(f"No QA data files found")
 
 print(f"Loading QA data from {data_files}...")
 # loadQM9BBBPdata
 qm9_data = []
 bbbp_data = []
 
 # loadQM9data
 if os.path.exists(qm9_file):
 print(f" Reading QM9 data from {qm9_file}...")
 with open(qm9_file, 'r', encoding='utf-8') as f:
 for line in f:
 line = line.strip()
 if not line:
 continue
 try:
 item = json.loads(line)
 if item.get("dataset", "").upper() == "QM9":
 qm9_data.append(item)
 except json.JSONDecodeError as e:
 print(f" ⚠️ Skipping invalid JSON line: {e}")
 continue
 print(f" Loaded {len(qm9_data)} QM9 samples")
 
 # loadBBBPdata
 if os.path.exists(bbbp_file):
 print(f" Reading BBBP data from {bbbp_file}...")
 with open(bbbp_file, 'r', encoding='utf-8') as f:
 for line in f:
 line = line.strip()
 if not line:
 continue
 try:
 item = json.loads(line)
 if item.get("dataset", "").upper() == "BBBP":
 bbbp_data.append(item)
 except json.JSONDecodeError as e:
 print(f" ⚠️ Skipping invalid JSON line: {e}")
 continue
 print(f" Loaded {len(bbbp_data)} BBBP samples")
 
 # QM9datasampleifmaxcount
 if qm9_max_samples is not None and len(qm9_data) > qm9_max_samples:
 print(f" Sampling QM9 data: {len(qm9_data)} -> {qm9_max_samples}")
 import random
 random.seed(seed)
 qm9_data = random.sample(qm9_data, qm9_max_samples)
 print(f" Sampled {len(qm9_data)} QM9 samples")
 
 # mergedata
 all_data = qm9_data + bbbp_data
 print(f" Total: {len(qm9_data)} QM9 + {len(bbbp_data)} BBBP = {len(all_data)} samples")
 
 # formatdata
 print("Formatting QA dataset...")
 processed_data = []
 meta_keys = [
 "id", "dataset", "source", "task_type", "smiles", "class_label",
 "property_name", "property_symbol", "property_description",
 "unit", "target_value", "all_targets"
 ]
 
 for item in all_data:
 user = safe_to_str(item.get("input", "")).strip()
 assistant = safe_to_str(item.get("output", "")).strip()
 # use Llama 3.2 format
 text = f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>"
 
 # buildformatdata
 formatted = {"text": text}
 for k in meta_keys:
 if k in item:
 formatted[k] = item[k]
 
 processed_data.append(formatted)
 
 # convertdatasetsformatprocess
 from datasets import Dataset
 processed = Dataset.from_list(processed_data)
 
 # filterinvaliddatacheck Llama 3.2 format
 def is_valid(example):
 t = example.get("text", "")
 # checkwhethercontains assistant markLlama 3.2 format
 has_assistant = "<|start_header_id|>assistant<|end_header_id|>" in t
 # formatifdata
 has_old_format = "<|assistant|>" in t
 return isinstance(t, str) and len(t.strip()) > 0 and (has_assistant or has_old_format)
 
 processed = processed.filter(is_valid, num_proc=1)
 
 # saveJSONLformat
 output_path = Path(output_file)
 output_path.parent.mkdir(parents=True, exist_ok=True)
 
 print(f"Saving to {output_file}...")
 with open(output_file, 'w', encoding='utf-8') as f:
 for item in processed:
 f.write(json.dumps(item, ensure_ascii=False) + '\n')
 
 print(f"✅ Preprocessed {len(processed)} QA samples to {output_file}")
 return len(processed)


def preprocess_sft_data(
 sft_file: str,
 output_file: str,
 filter_qm9: bool = True,
 target_size: Optional[int] = None,
 seed: int = 42,
):
 """
 processepochSFTdata
 
 Args:
 sft_file: SFTdatafilepath
 output_file: outputfilepath
 filter_qm9: whetherfilterQM9data
 target_size: targetdataSFT_DATA.jsontotal - firstepochdatafortotalconsistent
 seed: random
 
 data
 - SFT_DATA.json m data
 - firstepoch n dataQM9 + BBBP
 - epochshould (m - n) data
 - epochtotaldata = moriginalSFT_DATA.jsoncountconsistent
 """
 if not os.path.exists(sft_file):
 raise FileNotFoundError(f"SFT data file not found: {sft_file}")
 
 print(f"Loading SFT data from {sft_file}...")
 # readJSONfiletypeinconsistent
 if sft_file.endswith('.json'):
 # ifJSONfileformat
 with open(sft_file, 'r', encoding='utf-8') as f:
 raw_data = json.load(f)
 if not isinstance(raw_data, list):
 raw_data = [raw_data]
 else:
 # ifJSONLfile
 raw_data = []
 with open(sft_file, 'r', encoding='utf-8') as f:
 for line in f:
 line = line.strip()
 if not line:
 continue
 try:
 item = json.loads(line)
 raw_data.append(item)
 except json.JSONDecodeError as e:
 print(f" ⚠️ Skipping invalid JSON line: {e}")
 continue
 
 original_size = len(raw_data)
 print(f" Loaded {original_size} samples")
 
 # convertdatasetsformat
 from datasets import Dataset
 raw = Dataset.from_list(raw_data)
 
 # filterQM9data
 if filter_qm9:
 def filter_qm9_func(example):
 # checkdataset
 dataset = example.get("dataset", "")
 if "QM9" in str(dataset).upper():
 return False
 
 # checksource
 source = example.get("source", "")
 if "QM9" in str(source).upper():
 return False
 
 # checkid
 id_str = example.get("id", "")
 if "QM9" in str(id_str).upper():
 return False
 
 # checkmetadata.taske.g.: "qm9_property_query"
 metadata = example.get("metadata", {})
 if isinstance(metadata, dict):
 task = metadata.get("task", "")
 if "qm9" in str(task).lower():
 return False
 
 return True
 
 print("Filtering QM9 data...")
 filtered = raw.filter(filter_qm9_func, num_proc=min(4, os.cpu_count() or 1))
 print(f"Filtered: {len(raw)} -> {len(filtered)} (removed {len(raw) - len(filtered)} QM9 samples)")
 else:
 filtered = raw
 
 # data
 if target_size is not None:
 if len(filtered) > target_size:
 print(f"Balancing data: {len(filtered)} -> {target_size}")
 filtered = filtered.shuffle(seed=seed).select(range(target_size))
 elif len(filtered) < target_size:
 print(f"⚠️ Filtered data ({len(filtered)}) < target size ({target_size}), using all filtered data")
 else:
 print(f"Using all filtered data: {len(filtered)} samples (no size limit)")
 
 def format_sft_dataset(batch):
 texts = []
 for i in range(len(batch.get("input", []))):
 user = safe_to_str(batch["input"][i]).strip()
 assistant = safe_to_str(batch["output"][i]).strip()
 # use Llama 3.2 format
 texts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>")
 return {"text": texts}
 
 print("Formatting SFT dataset...")
 processed = filtered.map(
 format_sft_dataset,
 remove_columns=filtered.column_names,
 batched=True,
 batch_size=1000,
 num_proc=1,
 )
 
 # filterinvaliddatacheck Llama 3.2 format
 def is_valid(example):
 t = example.get("text", "")
 # checkwhethercontains assistant markLlama 3.2 format
 has_assistant = "<|start_header_id|>assistant<|end_header_id|>" in t
 # formatifdata
 has_old_format = "<|assistant|>" in t
 return isinstance(t, str) and len(t.strip()) > 0 and (has_assistant or has_old_format)
 
 processed = processed.filter(is_valid, num_proc=1)
 
 # saveJSONLformat
 output_path = Path(output_file)
 output_path.parent.mkdir(parents=True, exist_ok=True)
 
 print(f"Saving to {output_file}...")
 with open(output_file, 'w', encoding='utf-8') as f:
 for item in processed:
 f.write(json.dumps(item, ensure_ascii=False) + '\n')
 
 print(f"✅ Preprocessed {len(processed)} SFT samples to {output_file}")
 print(f" Original: {original_size}, Filtered: {len(filtered)}, Final: {len(processed)}")
 return len(processed)


def compute_qm9_stats(data_file: str) -> tuple:
 """computeQM9datavalue"""
 tasks = ["mu", "alpha", "homo", "lumo", "gap"]
 sums = [0.0] * len(tasks)
 sqs = [0.0] * len(tasks)
 cnt = 0
 
 print(f"Computing QM9 stats from {data_file}...")
 with open(data_file, 'r', encoding='utf-8') as f:
 for line in f:
 item = json.loads(line)
 if item.get("dataset") != "QM9" or item.get("task_type") != "regression":
 continue
 at = item.get("all_targets")
 if at is None:
 continue
 cnt += 1
 for i, t in enumerate(tasks):
 val = float(at.get(t, 0.0))
 sums[i] += val
 sqs[i] += val ** 2
 
 if cnt == 0:
 return None, None
 
 means = [s / cnt for s in sums]
 vars_ = [sq / cnt - m ** 2 for sq, m in zip(sqs, means)]
 stds = [max(1e-8, v) ** 0.5 for v in vars_]
 
 print(f"QM9 stats (from {cnt} samples):")
 print(f" Means: {means}")
 print(f" Stds: {stds}")
 
 return means, stds


def main():
 import argparse
 
 parser = argparse.ArgumentParser(description="Preprocess training data")
 parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
 parser.add_argument("--epoch", type=int, choices=[1, 2], required=True, help="Epoch number (1 or 2)")
 parser.add_argument("--output", type=str, help="Output file path (overrides config)")
 
 args = parser.parse_args()
 
 with open(args.config, 'r') as f:
 cfg = yaml.safe_load(f)
 
 if args.epoch == 1:
 # processfirstepochdata
 qa_data_dir = "${DATA_DIR:-/path/to/data}/MoleculeNet/qa_data"
 qm9_file = os.path.join(qa_data_dir, "qm9_qa_sft.jsonl")
 bbbp_file = os.path.join(qa_data_dir, "bbbp_qa_sft.jsonl")
 
 output_file = args.output or cfg.get("data", {}).get("epoch1_output", "data/epoch1_preprocessed.jsonl")
 
 # configreadQM9samplecountdefault60000
 qm9_max_samples = cfg.get("data", {}).get("epoch1_qm9_max_samples", 60000)
 seed = cfg.get("seed", 42)
 
 size = preprocess_qa_data(
 qm9_file, 
 bbbp_file, 
 output_file,
 qm9_max_samples=qm9_max_samples,
 seed=seed,
 )
 
 # computeQM9statistics
 qm9_means, qm9_stds = compute_qm9_stats(output_file)
 if qm9_means and qm9_stds:
 stats_file = output_file.replace(".jsonl", "_qm9_stats.json")
 with open(stats_file, 'w') as f:
 json.dump({"means": qm9_means, "stds": qm9_stds}, f, indent=2)
 print(f"✅ Saved QM9 stats to {stats_file}")
 
 print(f"\n✅ Epoch 1 preprocessing complete: {size} samples")
 
 else:
 # processepochdata
 sft_file = "${DATA_DIR:-/path/to/data}/SFT_data/SFT_DATA.json"
 
 # statisticsSFT_DATA.jsonQM9data
 # epoch2shoulduseSFT_DATA.jsontotal - QM9data
 if not os.path.exists(sft_file):
 raise FileNotFoundError(f"SFT data file not found: {sft_file}")
 
 print(f"Loading SFT data from {sft_file}...")
 # readJSONfile
 if sft_file.endswith('.json'):
 with open(sft_file, 'r', encoding='utf-8') as f:
 raw_data = json.load(f)
 if not isinstance(raw_data, list):
 raw_data = [raw_data]
 else:
 raw_data = []
 with open(sft_file, 'r', encoding='utf-8') as f:
 for line in f:
 line = line.strip()
 if not line:
 continue
 try:
 item = json.loads(line)
 raw_data.append(item)
 except json.JSONDecodeError as e:
 print(f" ⚠️ Skipping invalid JSON line: {e}")
 continue
 
 original_size = len(raw_data)
 print(f" Loaded {original_size} samples from SFT_DATA.json")
 
 # statisticsQM9data
 qm9_count = 0
 def is_qm9(example):
 # checkdataset
 dataset = example.get("dataset", "")
 if "QM9" in str(dataset).upper():
 return True
 
 # checksource
 source = example.get("source", "")
 if "QM9" in str(source).upper():
 return True
 
 # checkid
 id_str = example.get("id", "")
 if "QM9" in str(id_str).upper():
 return True
 
 # checkmetadata.taske.g.: "qm9_property_query"
 metadata = example.get("metadata", {})
 if isinstance(metadata, dict):
 task = metadata.get("task", "")
 if "qm9" in str(task).lower():
 return True
 
 return False
 
 for item in raw_data:
 if is_qm9(item):
 qm9_count += 1
 
 print(f" Found {qm9_count} QM9-related samples in SFT_DATA.json")
 
 # computetargetsizeepoch2shoulduse SFT_DATA.jsontotal - QM9data
 target_size = original_size - qm9_count
 print(f"Data balancing:")
 print(f" SFT_DATA.json total: {original_size} samples")
 print(f" QM9-related in SFT_DATA.json: {qm9_count} samples")
 print(f" Epoch 2 target: {target_size} samples (after filtering QM9)")
 
 output_file = args.output or cfg.get("data", {}).get("epoch2_output", "data/epoch2_preprocessed.jsonl")
 
 size = preprocess_sft_data(
 sft_file,
 output_file,
 filter_qm9=True,
 target_size=target_size,
 seed=cfg.get("seed", 42),
 )
 
 print(f"\n✅ Epoch 2 preprocessing complete: {size} samples")


if __name__ == "__main__":
 main()

