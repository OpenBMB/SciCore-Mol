"""
dataloadmodule
processdataloadformat
"""
import os
import json
import re
import hashlib
from typing import Optional, List, Dict, Any, Callable, Tuple
from datasets import load_dataset, Dataset
import torch


def safe_to_str(x):
 """convertstring"""
 if x is None:
 return ""
 if isinstance(x, (list, tuple)):
 return "\n".join(safe_to_str(xx) for xx in x)
 if isinstance(x, dict):
 return json.dumps(x, ensure_ascii=False)
 return str(x)

WORD_BOUNDARY_CHARS = set(" \n\t.,;:!?()[]{}")

# ==== mol span process ====

# “boundary”
# - span /continue
# - NOTE ** '.''['']' **
# [Na+].[Cl-]CCO.CS(=O)C /
_MOL_BOUNDARY_CHARS = " \n\t,;:!?{}"
# [] {} [Na+].[Cl-] 
_MOL_TRIM_CHARS = "'\"`“”‘’()"

_MOL_STOPWORDS = {"smiles", "Smiles", "SMILES", "logP", "NSAIDs"}

def _looks_like_molecule(span_text: str) -> bool:
 """
 rulejudge span “molecule”
 - length < 2
 - or SMILES / = # () [] @ + / -
 - otherwiseif >=4 toluene, ethanol, ibuprofen 
 rule
 """
 if not span_text:
 return False
 
 s = span_text.strip()
 if s in _MOL_STOPWORDS:
 return False
 if len(s) < 2:
 return False

 # SMILES / feature=#@+/-
 if any(c.isdigit() for c in s):
 return True
 if any(c in "=#()[]@+/-" for c in s):
 return True

 # if >=4 “”
 letters = [c for c in s if c.isalpha()]
 if len(letters) >= 4:
 return True

 return False


def _expand_and_merge_mol_spans(text: str, spans):
 """
 (start, end) spans process

 1) /“boundary”_MOL_BOUNDARY_CHARS
 2) /_MOL_TRIM_CHARS
 3) merge spans
 4) “molecule” span_looks_like_molecule

 Args:
 text: originalstringalreadyold <mol> label
 spans: List[(start, end)] token offset + prediction label=1

 Returns:
 List[(start, end)]process spanssortempty span
 """
 if not spans:
 return []

 expanded = []
 n = len(text)

 for s, e in spans:
 if s is None or e is None:
 continue
 if s >= e:
 continue

 # 1) boundary
 while s > 0 and text[s - 1] not in _MOL_BOUNDARY_CHARS:
 s -= 1

 # 2) boundary
 while e < n and text[e] not in _MOL_BOUNDARY_CHARS:
 e += 1

 # 3) 
 while s < e and text[s] in _MOL_TRIM_CHARS:
 s += 1
 while e > s and text[e - 1] in _MOL_TRIM_CHARS:
 e -= 1
 # 3.5) iflastempty / / end / token

 while s < e and text[e - 1] == '.':
 # e == nstring
 # e < n empty or or '<' <|eot_id|> 
 if e == n or text[e] in " \n\t<":
 e -= 1
 else:
 break

 if s < e:
 expanded.append((s, e))

 if not expanded:
 return []

 # 4) merge/ spans
 expanded.sort()
 merged = []
 for s, e in expanded:
 if not merged or s > merged[-1][1]:
 merged.append([s, e])
 else:
 merged[-1][1] = max(merged[-1][1], e)

 # 5) “molecule” spans
 final_spans = []
 for s, e in merged:
 span_text = text[s:e]
 if _looks_like_molecule(span_text):
 final_spans.append((s, e))

 return final_spans

def _expand_spans_to_word_boundaries(
 text: str,
 spans: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
 """
 span “boundary” MLP inferencescript
 - empty
 - empty
 span merge/
 """
 if not spans or not isinstance(text, str):
 return spans

 expanded: List[Tuple[int, int]] = []
 n = len(text)

 for start, end in spans:
 s, e = start, end
 #
 if s < 0:
 s = 0
 if e > n:
 e = n

 # “boundary”
 while s > 0 and text[s - 1] not in WORD_BOUNDARY_CHARS:
 s -= 1
 # “boundary”
 while e < n and text[e] not in WORD_BOUNDARY_CHARS:
 e += 1

 expanded.append((s, e))

 # span merge
 expanded.sort()
 merged: List[List[int]] = []
 for s, e in expanded:
 if not merged or s > merged[-1][1]:
 merged.append([s, e])
 else:
 merged[-1][1] = max(merged[-1][1], e)

 return [tuple(x) for x in merged]

def _save_dataset_to_jsonl(dataset: Dataset, file_path: str, is_tagged: bool = False):
 """interfacecallforcache"""
 os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
 with open(file_path, 'w', encoding='utf-8') as f:
 for example in dataset:
 f.write(json.dumps(example, ensure_ascii=False) + '\n')


def load_preprocessed_data(
 data_file: str,
 cache_dir: str = "./cache",
 use_cache: bool = True,
 max_samples: Optional[int] = None,
 max_message_chars: Optional[int] = None,
):
 """loadprocessdatausecacheretryreadsourcefile
 
 Args:
 data_file: originaldatapath
 max_message_chars: if messages totalsamplefilter
 """
 if not os.path.exists(data_file):
 raise FileNotFoundError(f"Data file not found: {data_file}")
 
 file_size = os.path.getsize(data_file)
 print(f"📂 Loading data from: {data_file} (size: {file_size / 1024 / 1024:.2f} MB)")
 if max_samples is not None:
 print(f"🔍 DEBUG MODE: Limiting to {max_samples} samples")
 
 data_list = []
 
 def normalize_content(content):
 """contentconvertstringformat"""
 if isinstance(content, str):
 return content
 if isinstance(content, list):
 text_parts = []
 for item in content:
 if isinstance(item, dict):
 if item.get("type") == "text":
 text_parts.append(str(item.get("text", "")))
 elif "text" in item:
 text_parts.append(str(item["text"]))
 elif isinstance(item, str):
 text_parts.append(item)
 return " ".join(text_parts) if text_parts else ""
 if isinstance(content, dict):
 if "text" in content:
 return str(content["text"])
 return json.dumps(content, ensure_ascii=False)
 return str(content) if content is not None else ""
 
 # read JSON / JSONL load_dataset retrycache
 if data_file.endswith('.jsonl'):
 with open(data_file, 'r', encoding='utf-8') as f:
 for line_num, line in enumerate(f, 1):
 line = line.strip()
 if not line:
 continue
 try:
 data = json.loads(line)
 if "messages" in data and isinstance(data["messages"], list):
 for msg in data["messages"]:
 if "content" in msg:
 msg["content"] = normalize_content(msg["content"])
 data_list.append(data)
 except json.JSONDecodeError as je:
 print(f"⚠️ Skipping invalid JSON at line {line_num}: {je}")
 except Exception as ex:
 print(f"⚠️ Error processing line {line_num}: {ex}")
 else:
 with open(data_file, 'r', encoding='utf-8') as f:
 try:
 loaded = json.load(f)
 except Exception as e:
 raise ValueError(f"Failed to load JSON file: {e}") from e
 if isinstance(loaded, list):
 data_iter = loaded
 else:
 data_iter = [loaded]
 for idx, data in enumerate(data_iter, 1):
 if isinstance(data, dict) and "messages" in data and isinstance(data["messages"], list):
 for msg in data["messages"]:
 if "content" in msg:
 msg["content"] = normalize_content(msg["content"])
 data_list.append(data)
 
 if not data_list:
 raise ValueError(f"No valid data loaded from {data_file}")
 
 try:
 raw = Dataset.from_list(data_list)
 except Exception as e2:
 print(f"⚠️ Dataset.from_list failed: {e2}")
 normalized_list = []
 for item in data_list:
 normalized_item = {}
 for k, v in item.items():
 if k == "messages" and isinstance(v, list):
 normalized_item[k] = v
 elif isinstance(v, (dict, list)) and k != "messages":
 normalized_item[k] = json.dumps(v, ensure_ascii=False)
 else:
 normalized_item[k] = v
 normalized_list.append(normalized_item)
 raw = Dataset.from_list(normalized_list)
 
 print(f"📊 Loaded {len(raw)} raw samples")
 
 if max_samples is not None and len(raw) > max_samples:
 raw = raw.select(range(max_samples))
 
 def _parse_chatml_text_to_messages(text: str):
 """
 ChatML/Qwen text:
 <|im_start|>user\n...\n<|im_end|>\n<|im_start|>assistant\n...\n<|im_end|>\n
 parse messages=[{role, content}, ...]
 """
 if not isinstance(text, str) or not text.strip():
 return None

 pattern = r"<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>"
 matches = re.findall(pattern, text, flags=re.DOTALL)
 if not matches:
 return None

 msgs = []
 for role, content in matches:
 msgs.append({"role": role, "content": content.strip()})

 # user+assistant SFTotherwise
 has_user = any(m["role"] == "user" and m["content"] for m in msgs)
 has_asst = any(m["role"] == "assistant" and m["content"] for m in msgs)
 if not (has_user and has_asst):
 return None
 return msgs

 def check_and_preserve_messages(example):
 """
 input
 1) messages formatcurrent loader defaultformat
 2) textChatML/Qwen parse messages

 target
 - example["messages"] list[dict] user+assistant
 - text "__MESSAGES_PLACEHOLDER__" load_training_data tokenizer.apply_chat_template generate text
 """
 msgs = example.get("messages", None)
 text = example.get("text", "")

 # -------- case 1: messages --------
 if isinstance(msgs, list):
 has_valid_content = False
 for msg in msgs:
 if not isinstance(msg, dict):
 continue
 content = msg.get("content", "")
 if not isinstance(content, str):
 content = json.dumps(content, ensure_ascii=False)
 msg["content"] = content
 if content and content.strip():
 has_valid_content = True

 example["text"] = "__MESSAGES_PLACEHOLDER__" if has_valid_content else ""
 return example

 # -------- case 2: messages text text+meta data--------
 if isinstance(text, str) and text.strip():
 parsed = _parse_chatml_text_to_messages(text)
 if parsed is not None:
 example["messages"] = parsed
 example["text"] = "__MESSAGES_PLACEHOLDER__"
 return example

 # ifparse ChatML raiseempty filter 
 example["text"] = ""
 return example

 # -------- case 3: /empty --------
 example["text"] = ""
 return example
 
 raw = raw.map(check_and_preserve_messages, num_proc=min(4, os.cpu_count() or 1))
 
 def is_valid(example):
 t = example.get("text", "")
 if t == "__MESSAGES_PLACEHOLDER__":
 return True
 return isinstance(t, str) and len(t.strip()) > 0
 
 processed = raw.filter(is_valid, num_proc=min(4, os.cpu_count() or 1))
 
 # filter messages total
 if max_message_chars is not None:
 def message_length_ok(example):
 msgs = example.get("messages", [])
 if not isinstance(msgs, list):
 return False
 total = 0
 for msg in msgs:
 if not isinstance(msg, dict):
 continue
 content = msg.get("content", "")
 if not isinstance(content, str):
 content = json.dumps(content, ensure_ascii=False)
 total += len(content)
 if total > max_message_chars:
 return False
 return True
 
 before = len(processed)
 processed = processed.filter(message_length_ok, num_proc=min(4, os.cpu_count() or 1))
 after = len(processed)
 print(f"✂️ Filtered long messages by max_message_chars={max_message_chars}: {before} -> {after}")
 
 print(f"✅ After filtering: {len(processed)} valid samples")
 
 if len(processed) == 0:
 raise ValueError(
 f"❌ No valid samples found in {data_file}!\n"
 f" Please check:\n"
 f" 1. Data file format (should be JSONL with 'text' field)\n"
 f" 2. Text field should not be empty"
 )
 
 def ensure_text_is_string(example):
 text = example.get("text", "")
 if not isinstance(text, str):
 if isinstance(text, list):
 example["text"] = text[0] if len(text) > 0 and isinstance(text[0], str) else ""
 else:
 example["text"] = str(text) if text is not None else ""
 else:
 example["text"] = text
 return example
 
 processed = processed.map(ensure_text_is_string, num_proc=16)
 
 return processed


def format_dataset_with_offline_spans(
 batch: Dict[str, List],
 tag_text_with_classifier: Optional[Callable[[str], str]] = None,
) -> Dict[str, List]:
 """formatdataoptionaluse"""
 texts = []
 inputs = batch.get("input", [])
 outputs = batch.get("output", [])
 
 for i in range(len(inputs)):
 user = safe_to_str(inputs[i]).strip()
 assistant = safe_to_str(outputs[i]).strip()
 # use "User / Assistant" formatmodel token
 # chat_template tokenizer training/inferencedecide
 concat = f"User: {user}\n\nAssistant: {assistant}"
 
 if tag_text_with_classifier is not None:
 tagged = tag_text_with_classifier(concat)
 texts.append(tagged)
 else:
 texts.append(concat)
 
 result = {"text": texts}
 
 # metaif
 meta_keys = [
 "id", "dataset", "source", "task_type", "smiles", "class_label",
 "property_name", "property_symbol", "property_description",
 "unit", "target_value", "all_targets"
 ]
 for k in meta_keys:
 if k in batch:
 result[k] = batch[k]
 
 return result


def create_tag_text_function(
 tokenizer,
 llm,
 offline_token_head,
 local_rank: int,
 max_length: int = 512,
) -> Optional[Callable[[str], str]]:
 """createfunctionversionfor"""
 if offline_token_head is None:
 return None
 
 def tag_text_with_classifier(text: str) -> str:
 if not isinstance(text, str) or not text:
 return text
 try:
 # clear <mol> label
 clean = re.sub(r"</?mol>", "", text)
 enc = tokenizer(
 clean,
 return_tensors="pt",
 return_offsets_mapping=True,
 truncation=True,
 max_length=max_length,
 padding=False,
 )
 input_ids = enc["input_ids"].to(local_rank)
 attn = enc["attention_mask"].to(local_rank)
 offsets = enc["offset_mapping"][0].tolist()
 
 with torch.no_grad():
 out = llm(
 input_ids=input_ids,
 attention_mask=attn,
 output_hidden_states=True,
 return_dict=True
 )
 hs = out.hidden_states[-1] # (1, T, H)
 # dtype matchget offline_token_head dtype
 try:
 head_dtype = next(offline_token_head.parameters()).dtype
 if head_dtype != hs.dtype:
 hs = hs.to(head_dtype)
 except (StopIteration, AttributeError):
 # ifparametergetdtypeusedefaultfloat32
 if hs.dtype != torch.float32:
 hs = hs.to(torch.float32)
 logits = offline_token_head(hs) # (1, T, 2)
 preds = torch.argmax(logits, dim=-1)[0].tolist()
 
 # span token offset 
 spans = []
 cur = None
 for p, (s, e) in zip(preds, offsets):
 if s == e:
 continue
 if p == 1:
 if cur is None:
 cur = [s, e]
 else:
 cur[1] = e
 else:
 if cur is not None:
 spans.append(tuple(cur))
 cur = None
 if cur is not None:
 spans.append(tuple(cur))

 if not spans:
 return clean
 
 # rule + merge + + filter
 spans = _expand_and_merge_mol_spans(clean, spans)
 if not spans:
 return clean
 
 # === special token ===
 # insert <mol> label
 # dependencymodel Llama header token
 special_tokens = [
 "<mol>", "</mol>",
 ]
 
 # all token 
 special_token_ranges = []
 for st in special_tokens:
 start = 0
 while True:
 pos = clean.find(st, start)
 if pos == -1:
 break
 special_token_ranges.append((pos, pos + len(st)))
 start = pos + 1
 
 # check token range <|start_header_id|>...<|end_header_id|>
 header_pairs = []
 start_pos = 0
 while True:
 start_header = clean.find("<|start_header_id|>", start_pos)
 if start_header == -1:
 break
 end_header = clean.find("<|end_header_id|>", start_header)
 if end_header != -1:
 header_pairs.append((start_header, end_header + len("<|end_header_id|>")))
 start_pos = end_header + len("<|end_header_id|>")
 else:
 break
 
 # filter token token spans
 filtered_spans = []
 for s, e in spans:
 is_special = False
 
 # checkwhether token 
 for st_start, st_end in special_token_ranges:
 if not (e <= st_start or s >= st_end):
 is_special = True
 break
 
 # checkwhether header includingboundary
 if not is_special:
 for pair_start, pair_end in header_pairs:
 if s >= pair_start and e <= pair_end:
 is_special = True
 break
 
 if not is_special:
 filtered_spans.append((s, e))
 
 if not filtered_spans:
 return clean
 
 # <mol></mol>index
 tagged = clean
 for s, e in reversed(filtered_spans):
 tagged = tagged[:e] + "</mol>" + tagged[e:]
 tagged = tagged[:s] + "<mol>" + tagged[s:]
 return tagged
 except Exception:
 # returnsoriginal
 return text
 
 return tag_text_with_classifier




def tag_text_with_smiles(text: str, smiles: Optional[str]) -> str:
 """
 based on SMILES match <mol></mol> label
 
 Args:
 text: original
 smiles: SMILES stringif Nonereturns
 
 Returns:
 
 """
 if not isinstance(text, str) or not text:
 return text
 
 if not smiles or not isinstance(smiles, str):
 return text
 
 # remove <mol> label
 clean_text = re.sub(r"</?mol>", "", text)
 
 # SMILES string
 # useregexmatchmatch SMILES
 # SMILES containsneeds
 escaped_smiles = re.escape(smiles)
 
 # allmatch
 matches = list(re.finditer(escaped_smiles, clean_text))
 
 if not matches:
 # ifmatchsize
 matches = list(re.finditer(re.escape(smiles), clean_text, re.IGNORECASE))
 
 if not matches:
 return clean_text
 
 # insertlabelindex
 tagged_text = clean_text
 for match in reversed(matches):
 start, end = match.span()
 # checkwhether token internal
 # token insertlabel
 special_tokens = [
 "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
 "<|user|>", "<|assistant|>", # format
 ]
 
 is_special = False
 for st in special_tokens:
 # checkmatchwhether token 
 st_start = clean_text.find(st, max(0, start - len(st)), min(len(clean_text), end + len(st)))
 if st_start != -1:
 st_end = st_start + len(st)
 # ifmatch token skip
 if not (end <= st_start or start >= st_end):
 is_special = True
 break
 
 if not is_special:
 # insertlabel
 tagged_text = tagged_text[:end] + "</mol>" + tagged_text[end:]
 tagged_text = tagged_text[:start] + "<mol>" + tagged_text[start:]
 
 return tagged_text


def create_batch_tag_text_function(
 tokenizer,
 llm,
 offline_token_head,
 local_rank: int,
 max_length: int = 512,
 batch_size: int = 32, # defaultvalue 32 value
) -> Optional[Callable[[List[str]], List[str]]]:
 """createfunction"""
 if offline_token_head is None:
 return None
 
 # LLM eval modememory
 original_training_mode = llm.training
 llm.eval()
 # saveoriginal use_cache set
 original_use_cache = None
 if hasattr(llm.config, 'use_cache'):
 original_use_cache = llm.config.use_cache
 llm.config.use_cache = False
 
 def tag_texts_batch(texts: List[str]) -> List[str]:
 """process"""
 if not texts:
 return texts
 
 results = []
 device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
 
 # process
 for i in range(0, len(texts), batch_size):
 batch_texts = texts[i:i + batch_size]
 # <mol> label
 batch_cleaned = [re.sub(r"</?mol>", "", t) if isinstance(t, str) else "" for t in batch_texts]
 
 try:
 # memory
 torch.cuda.empty_cache()
 
 # encodeusepadding
 enc = tokenizer(
 batch_cleaned,
 return_tensors="pt",
 return_offsets_mapping=True,
 truncation=True,
 max_length=max_length,
 padding=True,
 )
 input_ids = enc["input_ids"].to(device)
 attn = enc["attention_mask"].to(device)
 offsets_list = enc["offset_mapping"]
 
 # i.e. CPU encoderesult
 del enc
 
 with torch.no_grad():
 # usememoryinference
 out = llm(
 input_ids=input_ids,
 attention_mask=attn,
 output_hidden_states=True,
 return_dict=True,
 use_cache=False, # cachememory
 )
 hs = out.hidden_states[-1] # (B, T, H)
 # dtype match
 try:
 head_dtype = next(offline_token_head.parameters()).dtype
 if head_dtype != hs.dtype:
 hs = hs.to(head_dtype)
 except (StopIteration, AttributeError):
 if hs.dtype != torch.float32:
 hs = hs.to(torch.float32)
 logits = offline_token_head(hs) # (B, T, 2)
 preds = torch.argmax(logits, dim=-1).cpu().tolist() # (B, T)
 
 # GPU memory
 del out, hs, logits, input_ids, attn
 # offsets_list CPU
 offsets_list_cpu = offsets_list
 torch.cuda.empty_cache()
 
 # eachsampleprocess
 for j, (clean_text, pred, offsets) in enumerate(zip(batch_cleaned, preds, offsets_list_cpu)):
 if not clean_text:
 results.append(batch_texts[j])
 continue
 
 # spantoken offset layer
 spans = []
 cur = None
 offsets_items = offsets if isinstance(offsets, list) else offsets.tolist()
 for p, (s, e) in zip(pred, offsets_items):
 if s == e:
 continue
 if p == 1:
 if cur is None:
 cur = [s, e]
 else:
 cur[1] = e
 else:
 if cur is not None:
 spans.append(tuple(cur))
 cur = None
 if cur is not None:
 spans.append(tuple(cur))
 
 if not spans:
 results.append(clean_text)
 continue
 
 # + merge + + filter
 spans = _expand_and_merge_mol_spans(clean_text, spans)
 if not spans:
 results.append(clean_text)
 continue
 
 # === special token sampleversionconsistent ===
 special_tokens = [
 "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
 "<|user|>", "<|assistant|>", # format
 ]
 
 # all token 
 special_token_ranges = []
 for st in special_tokens:
 start = 0
 while True:
 pos = clean_text.find(st, start)
 if pos == -1:
 break
 special_token_ranges.append((pos, pos + len(st)))
 start = pos + 1
 
 # check token range <|start_header_id|>...<|end_header_id|>
 header_pairs = []
 start_pos = 0
 while True:
 start_header = clean_text.find("<|start_header_id|>", start_pos)
 if start_header == -1:
 break
 end_header = clean_text.find("<|end_header_id|>", start_header)
 if end_header != -1:
 header_pairs.append((start_header, end_header + len("<|end_header_id|>")))
 start_pos = end_header + len("<|end_header_id|>")
 else:
 break
 
 # filter token token spans
 filtered_spans = []
 for s, e in spans:
 is_special = False
 
 # checkwhether token 
 for st_start, st_end in special_token_ranges:
 if not (e <= st_start or s >= st_end):
 is_special = True
 break
 
 # checkwhether header includingboundary
 if not is_special:
 for pair_start, pair_end in header_pairs:
 if s >= pair_start and e <= pair_end:
 is_special = True
 break
 
 if not is_special:
 filtered_spans.append((s, e))
 
 if not filtered_spans:
 results.append(clean_text)
 continue
 
 tagged = clean_text
 for s, e in reversed(filtered_spans):
 tagged = tagged[:e] + "</mol>" + tagged[e:]
 tagged = tagged[:s] + "<mol>" + tagged[s:]
 results.append(tagged)
 
 # each batch successprocessmemory
 torch.cuda.empty_cache()
 
 except Exception as e:
 # checkwhethermemoryerror
 error_msg = str(e).lower()
 is_memory_error = any(keyword in error_msg for keyword in [
 "cuda out of memory",
 "out of memory",
 "cublas",
 "cudnn",
 "memory",
 ])
 
 if is_memory_error:
 # memoryerrorfallback
 print(f"❌ CUDA memory error during batch tagging (batch {i//batch_size}): {e}")
 print(f" Batch size: {batch_size}, Max length: {max_length}")
 print(f" Suggestion: Reduce offline_tagging_batch_size in config or reduce max_seq_length")
 # memory
 torch.cuda.empty_cache()
 raise RuntimeError(f"CUDA out of memory during offline tagging. Original error: {e}") from e
 else:
 # typeerrorfallback
 print(f"❌ Batch tagging failed for batch {i//batch_size}: {e}")
 torch.cuda.empty_cache()
 raise RuntimeError(f"Batch tagging failed. Original error: {e}") from e
 
 # restoreoriginalsetfunctionendno need to
 if original_training_mode:
 llm.train()
 if original_use_cache is not None:
 llm.config.use_cache = original_use_cache
 
 return results
 
 return tag_texts_batch




def load_training_data(
 cfg: Dict[str, Any],
 tokenizer,
 llm,
 offline_token_head: Optional[torch.nn.Module],
 local_rank: int,
) -> tuple:
 """
 loadtrainingdata
 
 Returns:
 train_dataset, eval_dataset
 """
 data_cfg = cfg.get("data", {})
 dataset_path = data_cfg.get("dataset_path") or cfg.get("train", {}).get("dataset_path")
 
 if not dataset_path:
 raise ValueError("dataset_path not found in config")
 
 # ifpathconvertpathdirectory
 if not os.path.isabs(dataset_path):
 # getdirectorytrain_sft.pydirectory
 code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 dataset_path = os.path.join(code_dir, dataset_path)
 dataset_path = os.path.abspath(dataset_path)
 
 print(f"📂 Using dataset path: {dataset_path}")
 
 # loadprocessdataalreadyformatcontainstext
 use_cache = cfg.get("data", {}).get("use_cache", True)
 # supportsdebugmodelimitdata seed randomsample
 max_samples = cfg.get("data", {}).get("debug_max_samples", None)
 max_tokens = cfg.get("data", {}).get("max_tokens", None) # token filtersample
 if max_samples is not None:
 print(f"🔍 DEBUG MODE ENABLED: max_samples={max_samples}")
 # NOTE max_samples load_preprocessed_datatotal N 
 max_message_chars = cfg.get("data", {}).get("max_message_chars", None)
 if max_message_chars is not None:
 print(f"⛔ Max message chars: {max_message_chars}")
 processed_dataset = load_preprocessed_data(
 dataset_path,
 cache_dir="./cache",
 use_cache=use_cache,
 max_samples=None,
 max_message_chars=max_message_chars,
 )

 # ifconfig debug_max_samplesdatavalueuse seed randomsample
 if max_samples is not None and len(processed_dataset) > max_samples:
 base_seed = int(cfg.get("seed", 42))
 rank = int(os.environ.get("RANK", 0))
 # different rank usedifferent seedmulti-GPUsampleconfig/
 shuffle_seed = base_seed + rank
 if rank == 0:
 print(f"🔀 Shuffling dataset with seed={shuffle_seed} and selecting first {max_samples} samples")
 processed_dataset = processed_dataset.shuffle(seed=shuffle_seed)
 processed_dataset = processed_dataset.select(range(max_samples))
 if rank == 0:
 print(f"✅ After debug sampling: {len(processed_dataset)} samples")
 
 # ifdatacontains messages use tokenizer.apply_chat_template convert text
 # stringusemodel chat template
 if len(processed_dataset) > 0 and "messages" in processed_dataset[0]:
 rank = int(os.environ.get("RANK", 0))
 if rank == 0:
 print("🔄 Converting messages format to text using tokenizer.apply_chat_template...")
 
 def convert_messages_with_template(example):
 """use tokenizer.apply_chat_template messages convert text
 example whether text messages chat_template 
 formatcurrent tokenizere.g. Mistral [INST] consistent
 """
 if "messages" in example and isinstance(example["messages"], list):
 messages = example["messages"]
 try:
 # use tokenizer chat_template generate
 formatted_text = tokenizer.apply_chat_template(
 messages,
 tokenize=False, # formatstring
 add_generation_prompt=False, # traininggenerateprompt/hint
 )
 example["text"] = formatted_text
 # optionalifmemorycandelete messages 
 # del example["messages"]
 except Exception as e:
 # if apply_chat_template fail User/Assistant 
 if rank == 0 and len(str(e)) < 200:
 print(f"⚠️ apply_chat_template failed for one sample: {e}, using fallback")
 text_parts = []
 for msg in messages:
 role = msg.get("role", "").lower()
 content = msg.get("content", "")
 if content:
 if role == "system":
 continue
 elif role == "user":
 text_parts.append(f"User: {content}")
 elif role == "assistant":
 text_parts.append(f"Assistant: {content}")
 example["text"] = "\n\n".join(text_parts) if text_parts else ""
 return example
 
 try:
 processed_dataset = processed_dataset.map(
 convert_messages_with_template,
 num_proc=min(4, os.cpu_count() or 1),
 desc="Converting messages to text with chat template"
 )
 if rank == 0:
 print("✅ Messages converted to text using chat template")
 except Exception as e:
 if rank == 0:
 print(f"⚠️ Failed to convert messages with template: {e}")
 print(" Using data as-is (may already have text field)")
 # ifconvertfailcontinueusedata
 
 # filter token samplebased ongenerate text
 if max_tokens is not None:
 rank = int(os.environ.get("RANK", 0))
 if rank == 0:
 print(f"✂️ Filtering samples longer than {max_tokens} tokens (tokenizer-based)")
 def token_length_ok(example):
 t = example.get("text", "")
 if not isinstance(t, str):
 return False
 # computelength
 ids = tokenizer.encode(t, add_special_tokens=True, truncation=False)
 return len(ids) <= max_tokens
 before = len(processed_dataset)
 processed_dataset = processed_dataset.filter(token_length_ok, num_proc=1)
 after = len(processed_dataset)
 if rank == 0:
 print(f"✂️ Token-length filter: {before} -> {after} samples (max_tokens={max_tokens})")
 
 # printdata
 rank = int(os.environ.get("RANK", 0))
 if rank == 0 and len(processed_dataset) > 0:
 print("\n" + "="*80)
 print("📋 First sample from dataset (after load_preprocessed_data):")
 print("="*80)
 first_sample = processed_dataset[0]
 print(f"Type: {type(first_sample)}")
 print(f"Content: {first_sample}")
 if isinstance(first_sample, dict):
 print(f"Keys: {list(first_sample.keys())}")
 for key, value in first_sample.items():
 print(f" {key}: type={type(value)}, value={str(value)[:200]}...")
 print("="*80 + "\n")
 
 # judgewhether epoch1viacheck dataset_path whethercontains "epoch1"
 is_epoch1 = "epoch1" in dataset_path.lower()
 
 # viadatajudgewhethercontainslabel
 has_mol_tags = False
 if len(processed_dataset) > 0:
 sample = processed_dataset[0]
 sample_text = sample.get("text", "")
 has_mol_tags = ("<mol>" in sample_text) and ("</mol>" in sample_text)
 if has_mol_tags:
 print(f"✅ Data contains <mol> tags, but cache metadata not found")

 # judgewhetherneeds
 is_already_tagged = has_mol_tags
 need_tagging = not is_already_tagged
 tagged_cache_file = None
 rank = int(os.environ.get("RANK", 0))

 if is_already_tagged:
 # datauseno need to
 if rank == 0:
 print("✅ Data is already tagged, skipping tagging step")
 need_tagging = False
 else:
 # datadeleteall <mol> / </mol> label
 if rank == 0:
 print("🧹 Removing any existing <mol></mol> tags before tagging")

 def strip_mol_tags(ex):
 try:
 t = ex.get("text", "")
 if isinstance(t, str):
 ex["text"] = re.sub(r"</?mol>", "", t)
 elif isinstance(t, list):
 # iflistprocesseach
 ex["text"] = [re.sub(r"</?mol>", "", str(item)) if isinstance(item, str) else str(item) for item in t]
 else:
 # typeconvertstringprocess
 ex["text"] = re.sub(r"</?mol>", "", str(t)) if t is not None else ""
 return ex
 except Exception as e:
 print(f"⚠️ Error in strip_mol_tags: {e}, example keys: {list(ex.keys()) if isinstance(ex, dict) else 'N/A'}")
 return ex

 try:
 processed_dataset = processed_dataset.map(
 strip_mol_tags,
 num_proc=min(16, os.cpu_count() or 1),
 desc="Stripping any existing <mol> tags",
 )
 if rank == 0:
 print(f"✅ Stripped <mol> tags, dataset size: {len(processed_dataset)}")
 except Exception as e:
 print(f"❌ Failed to strip <mol> tags: {e}")
 import traceback
 traceback.print_exc()
 raise
 
 # use tagged cache
 tagged_cache_file = None
 
 if not need_tagging:
 rank = int(os.environ.get("RANK", 0))
 if rank == 0:
 print("✅ Data already contains <mol> tags, skipping tagging")
 else:
 # epoch1use SMILES matchmethodno need to LLM inference
 if is_epoch1:
 rank = int(os.environ.get("RANK", 0))
 if rank == 0:
 print("🔄 Applying SMILES-based tagging for epoch1 data...")
 
 def tag_with_smiles(example):
 """use SMILES matchlabel"""
 text = example.get("text", "")
 smiles = example.get("smiles", None)
 if text and smiles:
 example["text"] = tag_text_with_smiles(text, smiles)
 return example
 
 # processuse mapcanparallel
 processed_dataset = processed_dataset.map(
 tag_with_smiles,
 num_proc=min(4, os.cpu_count() or 1),
 desc="Tagging with SMILES"
 )
 
 if rank == 0:
 print("✅ SMILES-based tagging completed")
 # savetaggeddatacachemark
 if use_cache and tagged_cache_file:
 print(f"💾 Saving tagged data to cache: {tagged_cache_file}")
 os.makedirs(os.path.dirname(tagged_cache_file) if os.path.dirname(tagged_cache_file) else ".", exist_ok=True)
 _save_dataset_to_jsonl(processed_dataset, tagged_cache_file, is_tagged=True)
 print(f"✅ Tagged cache saved ({len(processed_dataset)} samples, is_tagged=True)")
 
 # epoch2 useLLM + token classifier
 elif cfg.get("train", {}).get("use_offline_spans", False):
 if offline_token_head is None:
 if rank == 0:
 print("⚠️ use_offline_spans=True but offline_token_head is None")
 print(" This might happen if token classifier failed to load")
 print(" Will skip offline tagging and use data as-is")
 # if offline_token_head Noneskipusedata
 need_tagging = False
 else:
 # checkwhether DDP environment
 rank = int(os.environ.get("RANK", 0))
 world_size = int(os.environ.get("WORLD_SIZE", 1))
 is_distributed = world_size > 1
 # DDP environmentdataeachprocessprocesssync
 # use max_length for offline tagging memorycantraining max_seq_length
 training_max_length = cfg.get("train", {}).get("max_seq_length", 2048)
 # offline tagging uselengthmemorydefaulttraininglengthmin 512
 max_length = cfg.get("train", {}).get("offline_tagging_max_length", None)
 if max_length is None:
 max_length = max(512, training_max_length // 2) # defaultusetraininglengthmin 512
 batch_size = cfg.get("train", {}).get("offline_tagging_batch_size", 32)
 
 if is_distributed:
 # computeeachprocessdata
 total_size = len(processed_dataset)
 chunk_size = total_size // world_size
 start_idx = rank * chunk_size
 end_idx = start_idx + chunk_size if rank < world_size - 1 else total_size
 
 print(f"🔄 Applying offline tagging to add <mol> tags... (rank {rank}/{world_size-1}, processing samples {start_idx}-{end_idx-1})")
 print(f" Using max_length={max_length} (training max_length={training_max_length}), batch_size={batch_size}")
 
 # LLM eval modememory
 llm.eval()
 torch.cuda.empty_cache()
 
 # choosecurrentprocessdata
 processed_dataset_shard = processed_dataset.select(range(start_idx, end_idx))
 
 # useprocessfunctioninference
 batch_tag_func = create_batch_tag_text_function(
 tokenizer, llm, offline_token_head, local_rank, max_length, batch_size
 )
 
 if batch_tag_func is not None:
 def apply_tagging_batch(batch):
 """processinference"""
 texts = batch.get("text", [])
 if not texts:
 return batch
 
 # texts stringlist
 if isinstance(texts, list) and len(texts) > 0:
 # checkfirstwhetherstring
 if not isinstance(texts[0], str):
 # ifstringconvert
 texts = [str(t) if t is not None else "" for t in texts]
 elif not isinstance(texts, list):
 # iflistconvertlist
 texts = [str(texts)] if texts else []
 
 # process
 tagged_texts = batch_tag_func(texts)
 # returnslist
 if not isinstance(tagged_texts, list):
 tagged_texts = [tagged_texts] if tagged_texts else []
 batch["text"] = tagged_texts
 return batch
 
 print(f" Using batch size: {batch_size} (batch inference enabled)")
 processed_dataset_shard = processed_dataset_shard.map(
 apply_tagging_batch,
 batched=True,
 batch_size=batch_size,
 num_proc=1, # processCUDA
 )
 # GPU memory
 torch.cuda.empty_cache()
 print(f"✅ Offline tagging completed for shard {rank} ({len(processed_dataset_shard)} samples)")
 
 # syncallprocessallprocesscomplete
 import torch.distributed as dist
 if dist.is_initialized():
 dist.barrier()
 print(f"✅ All processes completed offline tagging (rank {rank})")
 
 # eachprocesssavefile rank 0 mergesavecache
 if use_cache and tagged_cache_file:
 # eachprocesssavefile
 shard_cache_file = tagged_cache_file.replace(".jsonl", f"_shard_{rank}.jsonl")
 os.makedirs(os.path.dirname(shard_cache_file) if os.path.dirname(shard_cache_file) else ".", exist_ok=True)
 _save_dataset_to_jsonl(processed_dataset_shard, shard_cache_file, is_tagged=True)
 print(f"💾 Rank {rank}: Saved shard to {shard_cache_file} ({len(processed_dataset_shard)} samples)")
 
 # syncallsave
 if dist.is_initialized():
 dist.barrier()
 
 # rank 0 allmergesave tagged cache
 if rank == 0:
 print(f"💾 Rank 0: Collecting all shards and saving to cache...")
 all_shards = []
 for r in range(world_size):
 shard_file = tagged_cache_file.replace(".jsonl", f"_shard_{r}.jsonl")
 if os.path.exists(shard_file):
 shard_dataset = load_dataset("json", data_files=shard_file, cache_dir="./cache", split="train", streaming=False)
 all_shards.append(shard_dataset)
 print(f" Loaded shard {r}: {len(shard_dataset)} samples")
 
 if all_shards:
 # mergeall
 from datasets import concatenate_datasets
 merged_dataset = concatenate_datasets(all_shards)
 print(f" Merged {len(merged_dataset)} samples from {len(all_shards)} shards")
 
 # save tagged cache
 _save_dataset_to_jsonl(merged_dataset, tagged_cache_file, is_tagged=True)
 print(f"✅ Tagged cache saved: {tagged_cache_file} ({len(merged_dataset)} samples, is_tagged=True)")
 
 # file
 for r in range(world_size):
 shard_file = tagged_cache_file.replace(".jsonl", f"_shard_{r}.jsonl")
 if os.path.exists(shard_file):
 try:
 os.remove(shard_file)
 meta_file = shard_file + ".meta"
 if os.path.exists(meta_file):
 os.remove(meta_file)
 except Exception as e:
 print(f"⚠️ Failed to remove shard file {shard_file}: {e}")
 
 # sync rank 0 completesave
 if dist.is_initialized():
 dist.barrier()
 
 # ifcachesaveallprocesscacheloaddata
 # eachprocessdatatrainingstepcorrect
 if os.path.exists(tagged_cache_file):
 print(f"📂 Reloading full tagged dataset from cache for all processes... (rank {rank})")
 cached_full = load_dataset("json", data_files=tagged_cache_file, cache_dir="./cache", split="train", streaming=False)
 print(f"✅ Loaded full dataset from cache: {len(cached_full)} samples (rank {rank})")
 processed_dataset = cached_full
 else:
 # ifcachesavefailusefallback
 print(f"⚠️ Cache file not found after saving, using shard for rank {rank} ({len(processed_dataset_shard)} samples)")
 processed_dataset = processed_dataset_shard
 else:
 # ifsavecacheuse
 print(f"✅ Using processed shard for rank {rank} ({len(processed_dataset_shard)} samples)")
 print(f" Note: Cache not saved, using shard. DataLoader will handle data distribution.")
 processed_dataset = processed_dataset_shard
 else:
 # processmodeprocessdata
 print(f"🔄 Applying offline tagging to add <mol> tags...")
 
 # LLM eval modememory
 llm.eval()
 torch.cuda.empty_cache()
 
 # useprocessfunctioninference
 batch_tag_func = create_batch_tag_text_function(
 tokenizer, llm, offline_token_head, local_rank, max_length, batch_size
 )
 
 if batch_tag_func is not None:
 def apply_tagging_batch(batch):
 """processinference"""
 texts = batch.get("text", [])
 if not texts:
 return batch
 
 # texts stringlist
 if isinstance(texts, list) and len(texts) > 0:
 # checkfirstwhetherstring
 if not isinstance(texts[0], str):
 # ifstringconvert
 texts = [str(t) if t is not None else "" for t in texts]
 elif not isinstance(texts, list):
 # iflistconvertlist
 texts = [str(texts)] if texts else []
 
 # process
 tagged_texts = batch_tag_func(texts)
 # returnslist
 if not isinstance(tagged_texts, list):
 tagged_texts = [tagged_texts] if tagged_texts else []
 batch["text"] = tagged_texts
 return batch
 
 print(f" Using batch size: {batch_size} (batch inference enabled)")
 processed_dataset = processed_dataset.map(
 apply_tagging_batch,
 batched=True,
 batch_size=batch_size,
 num_proc=1, # processCUDA
 )
 print("✅ Offline tagging completed")
 # savetaggeddatacachemark
 if use_cache and tagged_cache_file:
 print(f"💾 Saving tagged data to cache: {tagged_cache_file}")
 os.makedirs(os.path.dirname(tagged_cache_file) if os.path.dirname(tagged_cache_file) else ".", exist_ok=True)
 _save_dataset_to_jsonl(processed_dataset, tagged_cache_file, is_tagged=True)
 print(f"✅ Tagged cache saved ({len(processed_dataset)} samples, is_tagged=True)")
 
 # trainingvalidate
 eval_split = cfg.get("train", {}).get("eval_split", 0.05)
 split = processed_dataset.train_test_split(
 test_size=eval_split,
 seed=cfg.get("seed", 42)
 )
 
 train_size = len(split["train"])
 eval_size = len(split["test"])
 print(f"📈 Dataset split: {train_size} train, {eval_size} eval (split={eval_split})")
 
 if train_size == 0:
 raise ValueError(
 f"❌ Training dataset is empty after splitting!\n"
 f" Total samples: {len(processed_dataset)}\n"
 f" Eval split: {eval_split}\n"
 f" This might happen if eval_split is too large or dataset is too small"
 )
 
 # printreturnstrain_datasetdata
 rank = int(os.environ.get("RANK", 0))
 if rank == 0 and len(split["train"]) > 0:
 print("\n" + "="*80)
 print("📋 First sample from train_dataset (final, before return):")
 print("="*80)
 first_train_sample = split["train"][0]
 print(f"Type: {type(first_train_sample)}")
 print(f"Content: {first_train_sample}")
 if isinstance(first_train_sample, dict):
 print(f"Keys: {list(first_train_sample.keys())}")
 for key, value in first_train_sample.items():
 if key == "text":
 print(f" {key}: type={type(value)}, length={len(str(value))}, preview={str(value)[:200]}...")
 else:
 print(f" {key}: type={type(value)}, value={str(value)[:200]}...")
 print("="*80 + "\n")
 
 return split["train"], split["test"]


def compute_qm9_stats_from_dataset(dataset) -> tuple:
 """datacomputeQM9statistics"""
 tasks = ["mu", "alpha", "homo", "lumo", "gap"]
 sums = [0.0] * len(tasks)
 sqs = [0.0] * len(tasks)
 cnt = 0
 
 for ex in dataset:
 if ex.get("dataset") != "QM9" or ex.get("task_type") != "regression":
 continue
 at = ex.get("all_targets")
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
 
 return means, stds


def clean_cached_data(cache_file: str, output_file: Optional[str] = None):
 """
 cachedataerror
 
 fix
 1. remove token <mol> label <|start_header_id|><mol>assistant</mol><|end_header_id|>
 2. "the question is" "the answer is" prefixconvertformat
 
 Args:
 cache_file: cachefilepath
 output_file: outputfilepathif Nonefile
 """
 if not os.path.exists(cache_file):
 print(f"❌ Cache file not found: {cache_file}")
 return

 print(f"📂 Loading cache file: {cache_file}")
 dataset = load_dataset("json", data_files=cache_file, cache_dir="./cache", split="train", streaming=False)
 print(f"📊 Loaded {len(dataset)} samples")
 
 def clean_text(text: str) -> str:
 """error"""
 if not isinstance(text, str):
 return text
 
 # 1. remove token <mol> labelold Llama 3.2 formatcache
 # fix <|start_header_id|><mol>assistant</mol><|end_header_id|> -> <|start_header_id|>assistant<|end_header_id|>
 if "<|start_header_id|>" in text and "<|end_header_id|>" in text:
 text = re.sub(
 r'<\|start_header_id\|><mol>(assistant|user)</mol><\|end_header_id\|>',
 r'<|start_header_id|>\1<|end_header_id|>',
 text
 )
 
 # 2. "the question is" "the answer is" prefix
 # ifalreadycontainsformatprefixneedsremoveprefix
 if "<|start_header_id|>assistant<|end_header_id|>" in text:
 # ifalreadyformat assistant "the question is" "the answer is"
 # extract assistant 
 assistant_match = re.search(
 r'<\|start_header_id\|>assistant<\|end_header_id\|>\s*\n\s*\n(.*?)(?:\s*<\|eot_id\|>|$)',
 text,
 re.DOTALL
 )
 if assistant_match:
 assistant_content = assistant_match.group(1)
 # checkwhethercontains "the question is" "the answer is"
 pattern = r'the\s+question\s+is\s+(.+?)\s*,\s*the\s+answer\s+is\s+(.+?)(?:\s*<\|eot_id\|>|$)'
 match = re.search(pattern, assistant_content, re.IGNORECASE | re.DOTALL)
 if match:
 # extract answer ignore question question already user 
 answer = match.group(2).strip()
 # replace assistant remove "the question is ... , the answer is" prefix answer
 # usestringreplaceregex
 start_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
 end_marker = "<|eot_id|>"
 start_idx = text.find(start_marker)
 if start_idx != -1:
 start_idx += len(start_marker)
 end_idx = text.find(end_marker, start_idx)
 if end_idx != -1:
 # replace assistant 
 text = text[:start_idx] + answer + text[end_idx:]
 else:
 # ifformat "the question is" "the answer is" buildformat
 pattern = r'the\s+question\s+is\s+(.+?)\s*,\s*the\s+answer\s+is\s+(.+?)(?:\s*<\|eot_id\|>|$)'
 match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
 if match:
 question = match.group(1).strip()
 answer = match.group(2).strip()
 # format Llama 3.2 format
 text = f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
 
 return text
 
 def clean_example(example: Dict[str, Any]) -> Dict[str, Any]:
 """sample"""
 if "text" in example:
 example["text"] = clean_text(example["text"])
 return example
 
 print("🧹 Cleaning cached data...")
 cleaned_dataset = dataset.map(clean_example, num_proc=min(4, os.cpu_count() or 1))
 
 # savedata
 if output_file is None:
 output_file = cache_file
 
 print(f"💾 Saving cleaned data to: {output_file}")
 _save_dataset_to_jsonl(cleaned_dataset, output_file, is_tagged=True)
 print(f"✅ Cleaned cache saved ({len(cleaned_dataset)} samples)")


if __name__ == "__main__":
 """
 cachedata
 
 
 python -m modules.data_loader <cache_file> [output_file]
 """
 import sys
 if len(sys.argv) < 2:
 print("Usage: python -m modules.data_loader <cache_file> [output_file]")
 print("Example: python -m modules.data_loader ./cache/epoch2_preprocessed_tagged_offline_fa392044.jsonl")
 sys.exit(1)
 
 cache_file = sys.argv[1]
 output_file = sys.argv[2] if len(sys.argv) > 2 else None
 clean_cached_data(cache_file, output_file)

