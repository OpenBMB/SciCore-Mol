# mol_aware_lm_integrated.py
# -*- coding: utf-8 -*-
import os
import json
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
import logging

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from .gnn import GVPEncoder
from .mlp import MLPAdapter
from .tools import extract_and_convert_online

# use LDMol moleculegenerate
# BUG: diffusion fallback bug, needs
ENABLE_DIFFUSION_FALLBACK = False
from .ldmol_component import LDMolInferer

# use Layer2 reactionyieldprediction
from .layer2_component import Layer2Inferer

# RDKit
from rdkit import Chem

# log
import sys
import io
import os

# stdoutstderruseUTF-8encode
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

# ifstdout/stderrUTF-8
if hasattr(sys.stdout, 'buffer') and (not hasattr(sys.stdout, 'encoding') or sys.stdout.encoding != 'utf-8'):
 try:
 sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
 except (AttributeError, ValueError):
 pass
if hasattr(sys.stderr, 'buffer') and (not hasattr(sys.stderr, 'encoding') or sys.stderr.encoding != 'utf-8'):
 try:
 sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
 except (AttributeError, ValueError):
 pass

logging.getLogger("rdkit").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# configlogginguseUTF-8encode
class UTF8StreamHandler(logging.StreamHandler):
 """logoutputuseUTF-8encodeStreamHandler"""
 def __init__(self, stream=None):
 if stream is None:
 stream = sys.stderr
 super().__init__(stream)
 
 def emit(self, record):
 try:
 msg = self.format(record)
 stream = self.stream
 # useUTF-8encodewrite
 if hasattr(stream, 'buffer'):
 stream.buffer.write(msg.encode('utf-8', errors='replace'))
 stream.buffer.write(b'\n')
 self.flush()
 else:
 stream.write(msg)
 stream.write('\n')
 self.flush()
 except Exception:
 self.handleError(record)

# configloggingconfig
if not logging.root.handlers:
 logging.basicConfig(
 level=logging.INFO,
 format='%(asctime)s - %(levelname)s - %(message)s',
 handlers=[UTF8StreamHandler()]
 )

import torch.distributed as dist
import os, glob
import re

# data_loader.py _looks_like_molecule consistentjudge
_MOL_STOPWORDS = {"smiles", "Smiles", "SMILES", "logP", "NSAIDs"}

# setforjudgewhethershoulddetection
_BOUNDARY_CHARS = set(" \n\t,;:!?>")

def _is_boundary_token(tokenizer, token_id: int) -> bool:
 """
 judge token whetherempty
 detectiondetection
 """
 try:
 token_text = tokenizer.decode([token_id], skip_special_tokens=False)
 # check token whethercontainsor token 
 if not token_text:
 return False
 # if token or token /empty
 if any(c in _BOUNDARY_CHARS for c in token_text):
 return True
 # checkwhetherempty
 if token_text.strip() == "":
 return True
 return False
 except Exception:
 return False

def _looks_like_molecule(span_text: str) -> bool:
 """
 rulejudge span "molecule" data_loader.py consistent
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

 # if >=4 ""
 letters = [c for c in s if c.isalpha()]
 if len(letters) >= 4:
 return True

 return False

def has_hf_model_files(d: str) -> bool:
 if not os.path.isdir(d):
 return False
 # file / indexfile
 names = [
 "model.safetensors",
 "pytorch_model.bin",
 "model.safetensors.index.json",
 "pytorch_model.bin.index.json",
 "flax_model.msgpack",
 "tf_model.h5",
 ]
 if any(os.path.isfile(os.path.join(d, n)) for n in names):
 return True
 # filewhether index“directorycontainsweight”
 if glob.glob(os.path.join(d, "model-*-of-*.safetensors")):
 return True
 if glob.glob(os.path.join(d, "pytorch_model-*-of-*.bin")):
 return True
 return False

def any_rank_true(flag: bool) -> bool:
 """ rank Trueall rank True"""
 if not dist.is_available() or not dist.is_initialized():
 return flag
 t = torch.tensor([1 if flag else 0], device=torch.cuda.current_device())
 dist.all_reduce(t, op=dist.ReduceOp.MAX)
 return bool(t.item())

def zero_touch_module(module: torch.nn.Module) -> torch.Tensor:
 """ 0.0 * param.sum() module compute loss value"""
 if module is None:
 return torch.tensor(0.0, device=torch.cuda.current_device())
 z = torch.tensor(0.0, device=next(module.parameters()).device) if any(p.requires_grad for p in module.parameters()) else torch.tensor(0.0, device=torch.cuda.current_device())
 for p in module.parameters():
 if p.requires_grad:
 z = z + (0.0 * p.float().sum())
 return z

def build_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
 """
 position_idsvalid tokenmask=1padding 0
 LLM defineimplementi.e.
 """
 # (B, T)
 cumsum = attention_mask.long().cumsum(dim=1) * attention_mask.long()
 # 0 start 1
 pos_ids = (cumsum - attention_mask.long()).clamp(min=0)
 return pos_ids

class MolAwareCausalLM(nn.Module):
 """
 NER/GNN/Diffusion model <mol> vector“append”
 training labels=-100 inference KV token
 """
 # --------------------------- initialize ---------------------------
 def __init__(
 self,
 llm: nn.Module,
 tokenizer,
 mol_token: str = "<mol>",
 proxy: Optional[str] = None,
 debug: bool = False,
 target_layer_for_capture: int = -1,
 gvp_encoder_config: Optional[Dict] = None,
 mol_adapter_config: Optional[Dict] = None,
 diffusion_config: Optional[Dict] = None,
 diffusion_adapter_config: Optional[Dict] = None,
 token_classifier_head: Optional[nn.Module] = None,
 disable_gnn: bool = False, # whether GNN process
 use_ldmol: bool = True, # whetheruse LDMol @xyd
 ldmol_skip_text_encoder: bool = False, # whetherskip LDMol text_encoder @xyd
 layer2_config: Optional[Dict] = None, # Layer2 config
 use_layer2: bool = False, # whetheruse Layer2
 ):
 super().__init__()
 self.llm = llm
 self.tokenizer = tokenizer
 self.mol_token = mol_token
 self.mol_token_id = tokenizer.convert_tokens_to_ids(mol_token)
 self.pad_token_id = tokenizer.pad_token_id
 self.eos_token_id = tokenizer.eos_token_id
 self.debug = debug
 self.proxy = proxy
 self.disable_gnn = disable_gnn # GNN 

 # if self.mol_token_id is None or self.mol_token_id < 0:
 # raise ValueError(f"Tokenizer does not contain mol_token '{mol_token}'. Please add it first.")

 layers_ref = None
 if hasattr(self.llm, "model") and hasattr(self.llm.model, "layers"):
 layers_ref = self.llm.model.layers
 elif hasattr(self.llm, "transformer") and hasattr(self.llm.transformer, "h"):
 layers_ref = self.llm.transformer.h
 object.__setattr__(self, "_layers_ref", layers_ref)
 self.num_layers = len(self._layers_ref) if self._layers_ref is not None else 0
 self.target_layer_for_capture = (
 self.num_layers - 1 if (target_layer_for_capture < 0 and self.num_layers > 0) else target_layer_for_capture
 )
 self._capture_bucket: List[List[torch.Tensor]] = []
 self._capture_hook = None

 # ---------- component ----------
 try:
 llm_hidden_size = self.llm.config.hidden_size
 except Exception:
 llm_hidden_size = self.llm.config.text_config.hidden_size
 # GVPEncoder
 gvp_encoder_cfg = {
 "node_dims": (10, 1),
 "edge_dims": (1, 1),
 "hidden_scalar_dim": 256,
 "hidden_vector_dim": 16,
 "output_dim": 256,
 "num_layers": 4,
 }
 if gvp_encoder_config:
 gvp_encoder_cfg.update(gvp_encoder_config)

 # MLP Adapter GVP vectormapping LLM dimension
 mol_adapter_cfg = {
 "input_dim": gvp_encoder_cfg["output_dim"],
 "output_dim": llm_hidden_size,
 "hidden_dim": 2048,
 "num_layers": 2,
 }
 if mol_adapter_config:
 mol_adapter_cfg.update(mol_adapter_config)

 ##############################
 # LDMol initialize @xyd
 # via use_ldmol parameterwhetherload LDMolstage1 SFT trainingGPU memory
 # via ldmol_skip_text_encoder parameterwhetherskipload text_encoderinference Qwen

 # Temp
 self.use_ldmol = False
 # self.use_ldmol = use_ldmol 

 self.ldmol_skip_text_encoder = ldmol_skip_text_encoder
 
 if self.use_ldmol:
 self.ldmol = LDMolInferer(
 device=self._first_device(),
 skip_text_encoder=self.ldmol_skip_text_encoder,
 )
 else:
 self.ldmol = None
 logging.info(f"LDMol disabled (use_ldmol={use_ldmol})")
 self.enable_diffusion_fallback = ENABLE_DIFFUSION_FALLBACK and self.use_ldmol
 ##############################

 # initialize GVP encoderLayer2 needs
 self.gvp_encoder = GVPEncoder(**gvp_encoder_cfg).to(self._first_device())
 self.mol_adapter = MLPAdapter(**mol_adapter_cfg).to(self._first_device())
 
 ##############################
 # Layer2 initialize GVP encoder 
 # via use_layer2 parameterwhetherload Layer2forreactionyieldprediction
 self.use_layer2 = use_layer2
 if self.use_layer2:
 layer2_cfg = layer2_config or {}
 self.layer2_inferer = Layer2Inferer(
 config_path=layer2_cfg.get("config_path"),
 device=self._first_device(),
 gvp_encoder=self.gvp_encoder, # GVP encoder
 gvp_ckpt_path=layer2_cfg.get("gvp_ckpt_path"),
 )
 logging.info(f"Layer2 enabled")
 else:
 self.layer2_inferer = None
 logging.info(f"Layer2 disabled (use_layer2={use_layer2})")
 ##############################
 self.smiles_cache: Dict[str, str] = {}
 # optionalexternal token classheadfordetectionmolecule
 self.token_classifier_head = token_classifier_head
 

 # ---------- GNN Pipeline logstatistics ----------
 self.gnn_stats = {
 "smiles_processed": 0,
 "gnn_cache_hits": 0,
 "gnn_cache_misses": 0,
 "smiles_valid": 0,
 "smiles_invalid": 0,
 "diffusion_fallback_count": 0,
 "total_mol_embeddings": 0,
 }
 # Nsampleprintstatistics
 self.gnn_log_interval = 10

 # ---------- keyHF Trainer ----------
 # Trainer PreTrainedModel save
 self._config = getattr(self.llm, "config", None)
 self._keys_to_ignore_on_save = getattr(self.llm, "_keys_to_ignore_on_save", None)
 self._keys_to_ignore_on_load_missing = getattr(self.llm, "_keys_to_ignore_on_load_missing", None)
 self._keys_to_ignore_on_load_unexpected = getattr(self.llm, "_keys_to_ignore_on_load_unexpected", None)

 # --------------------------- HF interface ---------------------------
 @property
 def config(self):
 return self._config

 @property
 def _keys_to_ignore_on_save(self):
 return getattr(self.llm, "_keys_to_ignore_on_save", [])

 @_keys_to_ignore_on_save.setter
 def _keys_to_ignore_on_save(self, v):
 # AttributeErrorTrainer no need to llm 
 self.__dict__["__keys_to_ignore_on_save"] = v

 @property
 def _keys_to_ignore_on_load_missing(self):
 return getattr(self.llm, "_keys_to_ignore_on_load_missing", [])

 @_keys_to_ignore_on_load_missing.setter
 def _keys_to_ignore_on_load_missing(self, v):
 self.__dict__["__keys_to_ignore_on_load_missing"] = v

 @property
 def _keys_to_ignore_on_load_unexpected(self):
 return getattr(self.llm, "_keys_to_ignore_on_load_unexpected", [])

 @_keys_to_ignore_on_load_unexpected.setter
 def _keys_to_ignore_on_load_unexpected(self, v):
 self.__dict__["__keys_to_ignore_on_load_unexpected"] = v

 def to(self, *args, **kwargs):
 # sync LLM definemoduledevice
 super().to(*args, **kwargs)
 self.llm.to(*args, **kwargs)
 self.gvp_encoder.to(*args, **kwargs)
 self.mol_adapter.to(*args, **kwargs)
 if self.ldmol is not None:
 self.ldmol.to(*args, **kwargs)
 if self.layer2_inferer is not None:
 self.layer2_inferer.to(*args, **kwargs)
 return self

 # --------------------------- ---------------------------
 def _first_device(self):
 try:
 return self.llm.model.layers[0].input_layernorm.weight.device
 except Exception:
 return next(self.llm.parameters()).device

 def _get_smiles_from_context(self, llm_context: str) -> Optional[str]:
 if llm_context in self.smiles_cache:
 smiles_map = self.smiles_cache[llm_context]
 else:
 smiles_map = extract_and_convert_online(llm_context, self.proxy)
 self.smiles_cache[llm_context] = smiles_map
 if not smiles_map:
 return None
 last_cem = ""
 last_idx = -1
 for cem_name in smiles_map:
 idx = llm_context.rfind(cem_name)
 if idx > last_idx:
 last_idx = idx
 last_cem = cem_name
 return smiles_map.get(last_cem)

 def _extract_last_between_mol_tags(self, text: str) -> Optional[str]:
 """
 extract <mol>...</mol> internalreturns None
 """
 if not text:
 return None
 start = text.rfind("<mol>")
 end = text.rfind("</mol>")
 if start == -1 or end == -1 or end <= start:
 return None
 inner = text[start + len("<mol>"):end].strip()
 return inner if inner else None

 def _find_all_mol_spans(self, text: str):
 """
 returnsall <mol>...</mol> (inner_text, end_char_index) listend_char_index </mol> text 
 """
 if not text:
 return []
 try:
 spans = []
 for m in re.finditer(r"<mol>(.*?)</mol>", text, flags=re.DOTALL):
 inner = (m.group(1) or "").strip()
 spans.append((inner, m.end()))
 return spans
 except Exception:
 return []

 def _detect_mol_entities_with_classifier(self, input_ids: torch.Tensor, dec_text: str, enable_thinking: bool = False) -> List[Tuple[str, int]]:
 """
 use token_classifier_head detectionmolecule mlp_inference.py implement
 if token_classifier_head detectionfailfallback matchmethod
 
 Args:
 input_ids: input token ids
 dec_text: decode
 Returns:
 List[(inner_text, end_char_index)]: detectionmolecule spans
 """
 # contains <mol>...</mol>match
 if ("<mol>" in dec_text) and ("</mol>" in dec_text):
 return self._find_all_mol_spans(dec_text)
 # classmatch
 if self.token_classifier_head is None:
 # logger.info("[TokenClassifier] ❌ No token_classifier_head, using text matching fallback")
 return self._find_all_mol_spans(dec_text)
 
 # optimizationusetokencountjudge
 # converttoken
 try:
 # 4 = 1tokenSMILES
 estimated_tokens = len(dec_text) // 4
 max_tokens = getattr(self, '_max_text_length_for_detection', 4096) // 4 # converttoken
 if estimated_tokens > max_tokens * 2: # allowtolerance
 if getattr(self, '_verbose_logging', False):
 logging.debug(f"[TokenClassifier] ⚠️ Text too long (est. {estimated_tokens} tokens), will use truncation")
 except Exception:
 pass # iffailcontinueprocess
 
 try:
 # verbosemodelog
 if getattr(self, '_verbose_logging', False):
 text_preview = dec_text[:500] if len(dec_text) > 500 else dec_text
 preview_suffix = "..." if len(dec_text) > 500 else ""
 logger.info(f"[TokenClassifier] 🔍 Starting entity detection with classifier. Text length: {len(dec_text)} chars. Preview:\n{text_preview}{preview_suffix}")
 
 # 1) clearlabelforclassdetection
 text_clean = re.sub(r"</?mol>", "", dec_text)
 
 # 2) Tokenize clearlabelget offsets
 # optimizationusetokenizertruncationtokenizerprocesslengthlimit
 # usemax_lengthsupports2048 tokensprocess
 max_token_length = 2048
 _old_side = getattr(self.tokenizer, "truncation_side", "right")
 self.tokenizer.truncation_side = "left"
 try:
 enc = self.tokenizer(
 text_clean,
 return_tensors="pt",
 return_offsets_mapping=True,
 padding=False,
 truncation=True,
 max_length=max_token_length,
 add_special_tokens=False
 )
 finally:
 self.tokenizer.truncation_side = _old_side
 clean_input_ids = enc["input_ids"].to(input_ids.device)
 attention_mask = enc["attention_mask"].to(input_ids.device)
 offsets = enc["offset_mapping"][0].tolist()
 
 # 3) use LLM get hidden statesforclass
 device = input_ids.device
 with torch.no_grad():
 outputs = self.llm(
 input_ids=clean_input_ids,
 attention_mask=attention_mask,
 output_hidden_states=True,
 return_dict=True
 )
 hidden_states = outputs.hidden_states[-1] # (1, T, H)
 
 # 4) use token_classifier_head class
 with torch.no_grad():
 class_logits = self.token_classifier_head(hidden_states) # (1, T, 2)
 preds = torch.argmax(class_logits, dim=-1)[0].cpu().tolist()
 
 if getattr(self, '_verbose_logging', False):
 logger.info(f"[TokenClassifier] ✅ Classifier prediction completed. Found {sum(1 for p in preds if p == 1)} entity tokens")
 
 # 5) extract spansmergelabel1
 entity_spans = []
 current_start, current_end = None, None
 for label, (start, end) in zip(preds, offsets):
 if start == end:
 continue
 if label == 1: # moleculelabel
 if current_start is None:
 current_start, current_end = start, end
 else:
 current_end = end
 else:
 if current_start is not None:
 entity_spans.append((current_start, current_end))
 current_start, current_end = None, None
 if current_start is not None:
 entity_spans.append((current_start, current_end))
 
 # 6) processlabelemptyboundary
 expanded_spans = []
 for start, end in entity_spans:
 while start > 0 and text_clean[start-1] not in " \n\t.,;:!?()[]{}":
 start -= 1
 while end < len(text_clean) and text_clean[end] not in " \n\t.,;:!?()[]{}":
 end += 1
 expanded_spans.append((start, end))
 
 # 7) merge spanmark
 final_spans = []
 for span in sorted(expanded_spans):
 if not final_spans or span[0] > final_spans[-1][1]:
 final_spans.append(span)
 else:
 final_spans[-1] = (final_spans[-1][0], max(final_spans[-1][1], span[1]))
 
 # 8) convert (inner_text, end_char) format
 # NOTEend_char text_clean needsmappingoriginal dec_text
 result_spans = []
 for start, end in final_spans:
 inner_text = text_clean[start:end].strip()
 if inner_text:
 # original dec_text remove<mol>label
 # text_clean 
 idx_in_clean = text_clean.find(inner_text, max(0, start - 50), min(len(text_clean), end + 50))
 if idx_in_clean >= 0:
 # text_clean mapping dec_text<mol>label
 # returnsusevalue
 end_in_clean = idx_in_clean + len(inner_text)
 result_spans.append((inner_text, end_in_clean))
 
 if getattr(self, '_verbose_logging', False):
 if result_spans:
 logger.info(f"[TokenClassifier] 🎯 Detected {len(result_spans)} entities: {[r[0] for r in result_spans]}")
 else:
 logger.info("[TokenClassifier] ⚠️ No entities detected")
 
 return result_spans
 
 except Exception as e:
 logger.warning(f"[TokenClassifier] Failed to detect entities: {e}, falling back to text matching")
 return self._find_all_mol_spans(dec_text)


 def _decide_smiles_or_diffusion(self, llm_context_text: Optional[str], fallback_hctx: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
 """
 based on <mol>...</mol> internal
 - internal RDKit parse SMILES GVP -> mol_adapter
 - otherwiseallowuse diffusion path
 returnsmapping LLM dimensionvector None generate
 """
 inner = self._extract_last_between_mol_tags(llm_context_text or "")
 if inner:
 # RDKit whether SMILES
 is_smiles = False
 if Chem is not None:
 try:
 is_smiles = (Chem.MolFromSmiles(inner) is not None)
 except Exception:
 pass
 if is_smiles:
 try:
 if getattr(self, '_verbose_logging', False):
 logger.info(f"[GVP] 🔵 call GVP encoderSMILES: {inner[:100]}")
 gvp_embedding = self.gvp_encoder.forward_from_smiles(inner).squeeze(0)
 result = self.mol_adapter(gvp_embedding)
 if getattr(self, '_verbose_logging', False):
 logger.info(f"[GVP] ✅ GVP encoder completeembedding shape: {result.shape}")
 return result
 except Exception as e:
 if getattr(self, '_verbose_logging', False):
 logger.warning(f"[GVP] ❌ GVP encoder fail: {e}")
 return None
 # SMILES fail -> diffusion
 if self.enable_diffusion_fallback:
 if self._verbose_logging:
 logger.info(f"[Diffusion] 🟣 call Diffusion fallback: {inner[:100] if inner else 'None'}")
 result = self._generate_smiles_convert_to_embedding(text=llm_context_text)
 return result
 return None

 # label CEM -> SMILES
 smiles = self._get_smiles_from_context(llm_context_text or "") if llm_context_text else None
 if smiles:
 try:
 if getattr(self, '_verbose_logging', False):
 logger.info(f"[GVP] 🔵 call GVP encoderextractSMILES: {smiles[:100]}")
 gvp_embedding = self.gvp_encoder.forward_from_smiles(smiles).squeeze(0)
 result = self.mol_adapter(gvp_embedding)
 if getattr(self, '_verbose_logging', False):
 logger.info(f"[GVP] ✅ GVP encoder completeembedding shape: {result.shape}")
 return result
 except Exception as e:
 if getattr(self, '_verbose_logging', False):
 logger.warning(f"[GVP] ❌ GVP encoder fail: {e}")
 pass
 if fallback_hctx is not None:
 if getattr(self, '_verbose_logging', False):
 logger.info(f"[Diffusion] 🟣 call Diffusion fallbacklabel")
 result = self._black_box_from_hidden_hctx(fallback_hctx)
 if getattr(self, '_verbose_logging', False):
 logger.info(f"[Diffusion] ✅ Diffusion fallback completeembedding shape: {result.shape if result is not None else 'None'}")
 return result
 return None

 def _get_last_hidden_before_pos(self, row_ids: torch.Tensor, end_pos: int) -> torch.Tensor:
 assert end_pos > 0, "end_pos should be > 0"
 dev = self._first_device()
 prefix = row_ids[:end_pos].unsqueeze(0).to(dev)
 attn = (prefix != self.pad_token_id).long().to(dev)
 out = self.llm(input_ids=prefix, attention_mask=attn,
 output_hidden_states=True, use_cache=False, return_dict=True)
 return out.hidden_states[-1][0, -1, :].detach()
 
 def _generate_smiles_convert_to_embedding(self, text: str) -> Optional[torch.Tensor]:
 """
 useLDMoltextgeneratemoleculeSMILESconvertgvp embedding
 
 :param text: text for SMILES
 :type text: str
 :return: errorreturnsNone,returnsgvp embedding
 :rtype: Tensor | None
 """
 if self.ldmol is None or not LDMOL_AVAILABLE:
 logger.warning("LDMol unavailable, return None.")
 return None
 if self._verbose_logging:
 logger.info("[Diffusion] 🟣 start Diffusion generate")
 assert self.llm is not None and self.tokenizer is not None, "self.llm or self.tokenizer is None"
 generated_smiles = self.ldmol.generate_molecule(
 description=text,
 qwen=self.llm,
 qwen_tokenizer=self.tokenizer
 )
 if generated_smiles is None:
 logger.warning("LDMol fails to generate smiles, return None.")
 return None
 if self._verbose_logging:
 logger.info(f"✅ LDMol generate SMILES: {generated_smiles}")
 logger.info(f"[GVP] 🔵 call GVP encoderprocess Diffusion generate SMILES")
 gvp_embedding = self.gvp_encoder.forward_from_smiles(generated_smiles).squeeze(0)
 mol_emb = self.mol_adapter(gvp_embedding)
 if self._verbose_logging:
 logger.info(f"[GVP] ✅ GVP encoder completeembedding shape: {mol_emb.shape}")
 return mol_emb
 

 def _black_box_from_hidden_hctx(self, h_ctx: torch.Tensor) -> Optional[torch.Tensor]:
 """
 useLDMolLLMhidden stategeneratemoleculeSMILESconvertembedding
 """
 # TODO
 # raise ValueError("Updating @xyd")
 logger.info("[Diffusion] 🟣 start Diffusion generate hidden state")
 if self.ldmol_components is None or not LDMOL_AVAILABLE:
 logger.warning("[Diffusion] ❌ LDMol skip")
 return None
 
 dev = self._first_device()
 h_ctx = h_ctx.to(dev)
 
 try:
 # useLDMolLLM hidden stategenerateSMILES
 from .ldmol.inference import generate_molecule_from_llm_embedding
 gen_smiles = generate_molecule_from_llm_embedding(
 self.ldmol_components, h_ctx, dev
 )
 
 if not gen_smiles:
 if getattr(self, '_verbose_logging', False):
 logger.warning("[Diffusion] ❌ generatevalid SMILES")
 return None
 
 if getattr(self, '_verbose_logging', False):
 logger.info(f"[Diffusion] ✅ Diffusion generate SMILES: {gen_smiles}")
 
 # generateSMILESconvertembeddinguseGVP+mol_adapter
 if getattr(self, '_verbose_logging', False):
 logger.info(f"[GVP] 🔵 call GVP encoderprocess Diffusion generate SMILES")
 gvp_embedding = self.gvp_encoder.forward_from_smiles(gen_smiles).squeeze(0)
 mol_emb = self.mol_adapter(gvp_embedding)
 if getattr(self, '_verbose_logging', False):
 logger.info(f"[GVP] ✅ GVP encoder completeembedding shape: {mol_emb.shape}")
 return mol_emb
 
 except Exception as e:
 logger.warning(f"[BlackBox] ❌ LDMol generation failed: {e}")
 import traceback
 traceback.print_exc()
 return None

 # def _black_box_embed_offline(
 # self,
 # row_ids: torch.Tensor,
 # row_embeds: torch.Tensor,
 # row_mask: torch.Tensor,
 # pos_mol: int,
 # ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
 # # based on </mol> parse <mol>...</mol> internal
 # raise ValueError("TODO: _decide_smiles_or_diffusion interfaceuseneeds @xyd")
 # llm_context = self.tokenizer.decode(row_ids[:pos_mol + 1].tolist(), skip_special_tokens=True)
 # h_ctx = self._get_last_hidden_before_pos(row_ids, pos_mol) # [H]
 # emb = self._decide_smiles_or_diffusion(llm_context_text=llm_context, fallback_hctx=h_ctx)
 # return emb

 def _black_box_embed_online(
 self,
 llm_context_text: Optional[str] = None,
 context_ids: Optional[torch.Tensor] = None,
 h_ctx: Optional[torch.Tensor] = None,
 ) -> Optional[torch.Tensor]:
 if getattr(self, '_verbose_logging', False):
 logger.info(f"[Diffusion] 🟣 call Diffusion: {llm_context_text[:100] if llm_context_text else 'None'}...")
 if llm_context_text is not None:
 emb = self._decide_smiles_or_diffusion(llm_context_text=llm_context_text, fallback_hctx=h_ctx)
 if emb is not None:
 if getattr(self, '_verbose_logging', False):
 logger.info(f"[Diffusion] ✅ Diffusion completeembedding shape: {emb.shape}")
 return emb
 if context_ids is None and llm_context_text is not None:
 dev = self._first_device()
 toks = self.tokenizer(llm_context_text, return_tensors="pt", add_special_tokens=False)
 context_ids = toks["input_ids"].to(dev)
 if context_ids is not None:
 attn = (context_ids != self.pad_token_id).long().to(context_ids.device)
 out = self.llm(
 input_ids=context_ids, attention_mask=attn,
 output_hidden_states=True, use_cache=False, return_dict=True
 )
 h_ctx = out.hidden_states[-1][0, -1, :].detach()
 return self._black_box_from_hidden_hctx(h_ctx)
 return None

 # --------------------------- training/evaluation ---------------------------
 def forward(
 self,
 input_ids: Optional[torch.Tensor] = None,
 attention_mask: Optional[torch.Tensor] = None,
 labels: Optional[torch.Tensor] = None,
 **kwargs,
 ) -> CausalLMOutputWithPast:
 assert input_ids is not None, "MolAwareCausalLM needs input_ids"

 # 1) <mol> embedding <mol> append
 new_embeds, new_masks, new_labels, appended_mol_cnt = self._append_mol_embeds_to_end_offline(
 input_ids, attention_mask, labels
 )

 # 2) LLM 
 position_ids = build_position_ids(new_masks).to(new_masks.device)
 
 outputs = self.llm(
 inputs_embeds=new_embeds,
 attention_mask=new_masks,
 position_ids=position_ids,
 labels=new_labels,
 return_dict=True,
 **kwargs,
 )

 # 3) —— DDP process ——
 # " rank whetherappend mol vector"
 used_mol_local = (appended_mol_cnt > 0)
 # "all rank whether mol branch"
 used_mol_global = any_rank_true(used_mol_local)

 if used_mol_global and (not used_mol_local) and (outputs.loss is not None):
 if hasattr(self, "mol_adapter"):
 outputs.loss = outputs.loss + zero_touch_module(self.mol_adapter)
 if hasattr(self, "gnn_mlp"):
 outputs.loss = outputs.loss + zero_touch_module(self.gnn_mlp)
 if hasattr(self, "diffusion_mlp"):
 outputs.loss = outputs.loss + zero_touch_module(self.diffusion_mlp)

 return outputs

 def _append_mol_embeds_to_end_offline(
 self,
 input_ids: torch.Tensor,
 attention_mask: Optional[torch.Tensor],
 labels: Optional[torch.Tensor],
 ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
 """
 batch eachsample <mol>...</mol> based on </mol> “”
 - internalparse SMILESSMILES -> GVP -> mol_adapter LLM dimensionvector
 - otherwise diffusion pathbased on h_ctx vector
 vector“append” mask=1label=-100 LM loss
 returns(new_embeds, new_masks, new_labels, appended_mol_cnt_total)
 """
 # if GNNreturnsoriginal embeddingsprocess <mol> label
 if self.disable_gnn:
 embed_tokens = self.llm.get_input_embeddings()
 embeds = embed_tokens(input_ids)
 if attention_mask is None:
 attention_mask = (input_ids != self.pad_token_id).long().to(input_ids.device)
 return embeds, attention_mask, labels, 0
 
 assert input_ids.dim() == 2, "input_ids (B, T)"
 embed_tokens = self.llm.get_input_embeddings()
 emb_dev = embed_tokens.weight.device

 input_ids = input_ids.to(emb_dev)
 if attention_mask is not None:
 attention_mask = attention_mask.to(emb_dev)
 if labels is not None:
 labels = labels.to(emb_dev)

 B, T = input_ids.shape
 device = input_ids.device
 embeds = embed_tokens(input_ids) # (B, T, D)
 D = embeds.size(-1)

 if attention_mask is None:
 attention_mask = (input_ids != self.pad_token_id).long().to(device)
 has_labels = labels is not None

 rows_embeds, rows_masks, rows_labels = [], [], []
 max_len = 0
 appended_mol_cnt_total = 0 # batch append mol vector

 # per-forward localcacheSMILES -> mol_embcomputetraining mol_adapter
 per_forward_mol_emb_cache: Dict[str, torch.Tensor] = {}

 for b in range(B):
 row_ids = input_ids[b] # (T,)
 row_emb = embeds[b] # (T, D)
 row_msk = attention_mask[b] # (T,)
 row_lbl = labels[b] if has_labels else None

 # original token embed/mask/label 
 new_emb_list = [row_emb[i] for i in range(T)]
 new_msk_list = [int(row_msk[i].item()) for i in range(T)]
 new_lbl_list = [int(row_lbl[i].item()) for i in range(T)] if has_labels else None

 # use token_classifier_head detectionmolecule
 valid_len = int(row_msk.sum().item())
 dec_text = self.tokenizer.decode(row_ids[:valid_len].tolist(), skip_special_tokens=False)
 
 # get spansuse token_classifier_head fallback match
 spans = self._detect_mol_entities_with_classifier(row_ids[:valid_len], dec_text) # [(inner, end_char)]

 if spans:
 # use offsets_mapping mapping token index
 toks = self.tokenizer(dec_text, return_offsets_mapping=True, add_special_tokens=False)
 offsets = toks.get("offset_mapping")
 trigger_idx_to_span = {}
 if offsets is not None:
 # fast tokenizer: offsets batch 
 offsets = offsets[0].tolist() if hasattr(offsets, "tolist") else offsets
 # each span token boundaryindexfirst end>=end_char token
 for inner, end_char in spans:
 tok_idx = None
 for i_off, (_s, _e) in enumerate(offsets):
 if _e >= end_char and _e > 0:
 tok_idx = i_off
 break
 if tok_idx is None:
 tok_idx = len(offsets) - 1
 # validlength
 tok_idx = min(tok_idx, valid_len - 1)
 trigger_idx_to_span[tok_idx] = (inner, end_char)

 # traverse token indexsampleappendvector
 for trig_idx, (inner_text, end_char) in sorted(trigger_idx_to_span.items()):
 # padding skip
 if new_msk_list[trig_idx] == 0:
 continue

 mol_emb = None
 # whether SMILES
 is_smiles = False
 if Chem is not None and inner_text:
 try:
 is_smiles = (Chem.MolFromSmiles(inner_text) is not None)
 except Exception:
 is_smiles = False

 if is_smiles and inner_text:
 # logger.info(f"[EntityProcessing] ✅ Entity '{inner_text}' is valid SMILES, using GVP+adapter")
 # —— localcachecompute——
 if inner_text in per_forward_mol_emb_cache:
 mol_emb = per_forward_mol_emb_cache[inner_text]
 self.gnn_stats["gnn_cache_hits"] += 1
 else:
 # GVP freeze/frozen no_gradGPU memory/mol_adapter needsgradienttraining
 with torch.no_grad():
 gvp_embedding = self.gvp_encoder.forward_from_smiles(inner_text).squeeze(0)
 mol_emb = self.mol_adapter(gvp_embedding) # shape: [D]
 per_forward_mol_emb_cache[inner_text] = mol_emb
 self.gnn_stats["gnn_cache_misses"] += 1
 self.gnn_stats["smiles_processed"] += 1
 self.gnn_stats["smiles_valid"] += 1
 self.gnn_stats["total_mol_embeddings"] += 1
 elif self.enable_diffusion_fallback:
 if getattr(self, '_verbose_logging', False):
 logger.info(f"[EntityProcessing] 🎲 Entity '{inner_text}' is NOT valid SMILES, calling BLACKBOX fallback")
 # </mol> endcompute h_ctx / 
 ctx_text = dec_text[:end_char]
 mol_emb = self._black_box_embed_online(llm_context_text=ctx_text, context_ids=None, h_ctx=None)
 if mol_emb is not None:
 if getattr(self, '_verbose_logging', False):
 logger.info(f"[EntityProcessing] ✅ Blackbox returned embedding successfully")
 self.gnn_stats["diffusion_fallback_count"] += 1
 else:
 # keyblackbox failshouldtotalwarning
 logger.warning(f"[EntityProcessing] ❌ Blackbox returned None for entity '{inner_text[:50]}...'")
 else:
 if getattr(self, '_verbose_logging', False):
 logger.info(f"[EntityProcessing] ⚠️ Entity '{inner_text}' is invalid and fallback disabled")
 self.gnn_stats["smiles_processed"] += 1
 self.gnn_stats["smiles_invalid"] += 1

 if mol_emb is None:
 # keyifskipshouldrecord
 if getattr(self, "debug", False):
 if getattr(self, '_verbose_logging', False):
 logger.info("[Offline] Skip virtual step at </mol> (no embedding).")
 else:
 # verbose mode debug output
 pass
 continue

 new_emb_list.append(mol_emb)
 new_msk_list.append(1)
 if has_labels:
 new_lbl_list.append(-100)
 appended_mol_cnt_total += 1

 # sampleresult -> tensor
 new_len = len(new_msk_list)
 max_len = max(max_len, new_len)

 new_emb = torch.stack(new_emb_list, dim=0) # (L, D)
 new_msk = torch.tensor(new_msk_list, device=device, dtype=row_msk.dtype) # (L,)
 new_lbl = (torch.tensor(new_lbl_list, device=device, dtype=input_ids.dtype)
 if has_labels else None)

 rows_embeds.append(new_emb)
 rows_masks.append(new_msk)
 if has_labels:
 rows_labels.append(new_lbl)

 # alignmentlength padding
 padded_embeds, padded_masks = [], []
 padded_labels = [] if has_labels else None

 for b in range(B):
 E = rows_embeds[b]; M = rows_masks[b]
 pad_len = max_len - E.size(0)

 if pad_len > 0:
 E = torch.cat([E, torch.zeros(pad_len, D, device=E.device, dtype=E.dtype)], dim=0)
 M = torch.cat([M, torch.zeros(pad_len, device=M.device, dtype=M.dtype)], dim=0)
 if has_labels:
 L = rows_labels[b]
 L = torch.cat([L, torch.full((pad_len,), -100, device=L.device, dtype=L.dtype)], dim=0)
 else:
 L = None
 else:
 L = rows_labels[b] if has_labels else None

 padded_embeds.append(E.unsqueeze(0)) # (1, max_len, D)
 padded_masks.append(M.unsqueeze(0)) # (1, max_len)
 if has_labels:
 padded_labels.append(L.unsqueeze(0) if L is not None else None)

 new_embeds = torch.cat(padded_embeds, dim=0) # (B, max_len, D)
 new_masks = torch.cat(padded_masks, dim=0) # (B, max_len)
 new_labels = torch.cat(padded_labels, dim=0) if has_labels else None # (B, max_len) or None

 # keyifbatchprocessoutput
 if appended_mol_cnt_total > 0:
 if getattr(self, '_verbose_logging', False):
 logger.info(f"[Offline] Batch processed: appended {appended_mol_cnt_total} mol embeddings")
 # verbose modeoutput

 # printGNN pipelinestatisticskeytotaloutput
 if hasattr(self, "gnn_stats") and appended_mol_cnt_total > 0:
 stats = self.gnn_stats
 total = stats["gnn_cache_hits"] + stats["gnn_cache_misses"]
 if total > 0 and (stats["total_mol_embeddings"] % self.gnn_log_interval == 0):
 hit_rate = stats["gnn_cache_hits"] / total * 100 if total > 0 else 0
 if getattr(self, '_verbose_logging', False):
 # statistics
 logger.info(
 f"[GNN Pipeline] Stats: SMILES processed={stats['smiles_processed']}, "
 f"valid={stats['smiles_valid']}, invalid={stats['smiles_invalid']}, "
 f"cache_hits={stats['gnn_cache_hits']}, cache_misses={stats['gnn_cache_misses']}, "
 f"hit_rate={hit_rate:.1f}%, diffusion_fallback={stats['diffusion_fallback_count']}, "
 f"total_embeddings={stats['total_mol_embeddings']}"
 )
 else:
 # statisticskey
 logger.info(
 f"[GNN Pipeline] Processed {stats['total_mol_embeddings']} embeddings "
 f"(valid: {stats['smiles_valid']}, invalid: {stats['smiles_invalid']}, "
 f"cache hit rate: {hit_rate:.1f}%)"
 )
 
 if getattr(self, "debug", False):
 orig_tokens = attention_mask.sum().item()
 new_tokens = new_masks.sum().item()
 print(f"[MolAware/offline] appended {int(new_tokens - orig_tokens)} embeddings to batch end; "
 f"mol_appended_count={appended_mol_cnt_total}")

 return new_embeds, new_masks, new_labels, appended_mol_cnt_total


 @torch.no_grad()
 def generate(
 self,
 input_ids: Optional[torch.Tensor] = None,
 attention_mask: Optional[torch.Tensor] = None,
 realtime_mol: bool = True,
 max_new_tokens: int = 256,
 do_sample: bool = False,
 temperature: float = 1.0,
 top_k: int = 0,
 top_p: float = 1.0,
 eos_token_id: Optional[int] = None,
 repetition_penalty: float = 1.05,
 verbose_logging: bool = False, # logoutput
 max_text_length_for_detection: int = 4096, # lengthskipdetectionstopgeneratesupportsfew-shotprompt
 skip_special_tokens: bool = False,
 stop_on_eos: bool = True, # ✅ whetherEOS/eotstopdefault True
 **kwargs,
 ):
 """
 inferencestageprocessonline
 - token sample
 - boundary token detectionnew </mol> detectioninsert virtual stepinputs_embeds
 keyfix
 - ✅ restore stop tokensEOS / <|eot_id|>otherwiserun max_new_tokens
 - ✅ virtual step “(, end_char)”insert/
 """
 use_cache = kwargs.pop("use_cache", True)
 no_repeat_ngram_size = int(kwargs.pop("no_repeat_ngram_size", 0) or 0)

 try:
 self.llm.config.use_cache = True
 except Exception:
 pass

 if not realtime_mol:
 # HF generate EOS stop
 return self.llm.generate(
 input_ids=input_ids,
 attention_mask=attention_mask,
 max_new_tokens=max_new_tokens,
 do_sample=do_sample,
 temperature=temperature,
 top_k=top_k,
 top_p=top_p,
 eos_token_id=eos_token_id,
 repetition_penalty=repetition_penalty,
 use_cache=use_cache,
 **kwargs,
 )

 # realtime_mol supports input_ids / inputs_embeds
 # - ifuse input_ids token detectioninputs_embeds embedding insertclass GVP 
 # - if input_idsprocess
 # - if inputs_embedsskip token detectionalready embedding 
 inputs_embeds_extra = kwargs.pop("inputs_embeds", None)
 has_input_ids = input_ids is not None
 has_inputs_embeds = inputs_embeds_extra is not None
 
 if not has_input_ids and not has_inputs_embeds:
 raise ValueError("must input_ids inputs_embeds")
 
 if not realtime_mol:
 # realtime_mol modeuse generate
 if has_inputs_embeds and not has_input_ids:
 # inputs_embeds
 return self.llm.generate(
 inputs_embeds=inputs_embeds_extra,
 attention_mask=attention_mask,
 max_new_tokens=max_new_tokens,
 do_sample=do_sample,
 temperature=temperature,
 top_k=top_k,
 top_p=top_p,
 eos_token_id=eos_token_id,
 repetition_penalty=repetition_penalty,
 use_cache=use_cache,
 **kwargs,
 )
 else:
 # input_ids inputs_embeds generate supports
 return self.llm.generate(
 input_ids=input_ids,
 attention_mask=attention_mask,
 max_new_tokens=max_new_tokens,
 do_sample=do_sample,
 temperature=temperature,
 top_k=top_k,
 top_p=top_p,
 eos_token_id=eos_token_id,
 repetition_penalty=repetition_penalty,
 use_cache=use_cache,
 **kwargs,
 )
 
 # realtime_mol modesupports input_ids / inputs_embeds
 if has_input_ids:
 # input_idscan token detection
 if input_ids.size(0) > 1:
 raise ValueError(f"realtime_mol supports batch=1current batch={input_ids.size(0)}call batch_size=1 process")
 if attention_mask is None:
 attention_mask = (input_ids != self.pad_token_id).long()
 elif has_inputs_embeds:
 # inputs_embedsskip token detection
 if inputs_embeds_extra.size(0) > 1:
 raise ValueError(f"realtime_mol supports batch=1current batch={inputs_embeds_extra.size(0)}call batch_size=1 process")
 if attention_mask is None:
 # inputs_embeds infer attention_mask
 attention_mask = torch.ones(inputs_embeds_extra.size(0), inputs_embeds_extra.size(1), dtype=torch.long, device=inputs_embeds_extra.device)
 
 llm = self.llm
 dev = self._first_device()
 
 if has_input_ids:
 input_ids = input_ids.to(dev)
 if has_inputs_embeds:
 inputs_embeds_extra = inputs_embeds_extra.to(dev)
 attention_mask = attention_mask.to(dev)

 # setlogdetectionparameter
 self._verbose_logging = verbose_logging
 self._max_text_length_for_detection = max_text_length_for_detection
 self._detection_interval = getattr(self, "_detection_interval", 5) # Nboundarytokendetection
 self._boundary_token_count = 0

 # per-generation cacheSMILES -> mol_embinferencegradient
 gen_mol_emb_cache: Dict[str, torch.Tensor] = {}

 # ✅ key “” virtual step
 # (effective_text, end_char_pos) key
 processed_occurrences = set()

 # processed_pair_count / processed_inner_textsinsert
 processed_pair_count = 0
 processed_inner_texts = set()

 # ====== stop tokens✅ restore======
 stop_token_ids = set()
 end_id = eos_token_id if eos_token_id is not None else self.eos_token_id
 if (end_id is None or end_id < 0) and self.tokenizer is not None:
 # <|eot_id|>
 try:
 eot_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
 if eot_token_id is not None and eot_token_id >= 0:
 end_id = eot_token_id
 if verbose_logging:
 logger.info(f"[Generate] Using <|eot_id|> (token_id={end_id}) as EOS token fallback")
 except Exception:
 pass

 if stop_on_eos:
 if end_id is not None and end_id >= 0:
 stop_token_ids.add(int(end_id))
 # eotLlama end
 try:
 eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
 if eot is not None and eot >= 0:
 stop_token_ids.add(int(eot))
 except Exception:
 pass

 def _prepare_probs(_logits: torch.Tensor) -> torch.Tensor:
 probs = torch.softmax(_logits, dim=-1)
 probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
 probs = torch.clamp(probs, min=0.0)
 sum_probs = probs.sum(dim=-1, keepdim=True)
 probs = probs / sum_probs.clamp(min=1e-8)
 return probs

 def _apply_topk_topp_temp(_logits: torch.Tensor) -> torch.Tensor:
 logits2 = _logits
 if temperature and temperature != 1.0:
 logits2 = logits2 / float(temperature)
 if top_k and top_k > 0:
 v, _ = torch.topk(logits2, int(top_k))
 logits2 = logits2.masked_fill(logits2 < v[:, [-1]], float("-inf"))
 if top_p and top_p < 1.0:
 sorted_logits, sorted_indices = torch.sort(logits2, descending=True)
 probs = torch.softmax(sorted_logits, dim=-1)
 cumprobs = probs.cumsum(dim=-1)
 cutoff = (cumprobs > float(top_p)).float().cumsum(dim=-1).bool()
 sorted_logits[cutoff] = float("-inf")
 logits2 = torch.full_like(logits2, float("-inf")).scatter(1, sorted_indices, sorted_logits)
 return logits2

 def _apply_sampling(_logits: torch.Tensor) -> torch.Tensor:
 if do_sample:
 logits2 = _apply_topk_topp_temp(_logits)
 probs = _prepare_probs(logits2)
 if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs <= 0).all():
 return torch.argmax(_logits, dim=-1, keepdim=True)
 return torch.multinomial(probs, num_samples=1)
 return torch.argmax(_logits, dim=-1, keepdim=True)

 def _block_no_repeat_ngrams(logits: torch.Tensor, prefix_ids: List[int], gen_ids: List[int], n: int) -> torch.Tensor:
 """
 no_repeat_ngram_sizebased oncurrent prefix+generated
 transformers forbid“ ngram”
 """
 if n <= 0:
 return logits
 seq = prefix_ids + gen_ids
 if len(seq) < n:
 return logits
 # currentprediction len(seq) -> n-1 prefix
 prev_ngram = tuple(seq[-(n - 1):]) if n > 1 else tuple()
 # ngramprev_ngram -> next_token list
 banned = set()
 if n == 1:
 # n==1 forbid token
 return logits
 for i in range(len(seq) - n + 1):
 ng = tuple(seq[i:i + n])
 if ng[:-1] == prev_ngram:
 banned.add(ng[-1])
 if banned:
 logits[:, list(banned)] = float("-inf")
 return logits

 # ====== ======
 if has_input_ids:
 # use input_ids 
 outputs = llm(
 input_ids=input_ids,
 attention_mask=attention_mask,
 use_cache=use_cache,
 output_hidden_states=True,
 return_dict=True,
 **kwargs,
 )
 else:
 # inputs_embeds
 outputs = llm(
 inputs_embeds=inputs_embeds_extra,
 attention_mask=attention_mask,
 use_cache=use_cache,
 output_hidden_states=True,
 return_dict=True,
 **kwargs,
 )
 past = outputs.past_key_values
 attn_mask = attention_mask

 # ====== if inputs_embeds embedding insertclass GVP ======
 if has_input_ids and has_inputs_embeds:
 # input_ids inputs_embeds inputs_embeds insert
 model_dtype = next(self.llm.parameters()).dtype
 if inputs_embeds_extra.dtype != model_dtype:
 inputs_embeds_extra = inputs_embeds_extra.to(dtype=model_dtype)
 if inputs_embeds_extra.device != dev:
 inputs_embeds_extra = inputs_embeds_extra.to(device=dev)
 
 # inputs_embeds correct[1, seq_len, hidden_dim]
 if inputs_embeds_extra.dim() == 2:
 inputs_embeds_extra = inputs_embeds_extra.unsqueeze(0) # [seq_len, hidden_dim] -> [1, seq_len, hidden_dim]
 
 # insert virtual stepclass GVP 
 if verbose_logging:
 logger.info(f"[Generate] ✅ insert embeddingclass GVP : {inputs_embeds_extra.shape}")
 
 # update attention_mask embedding mask
 extra_seq_len = inputs_embeds_extra.size(1)
 extra_mask = torch.ones(1, extra_seq_len, device=dev, dtype=attn_mask.dtype)
 
 # insert embedding
 # NOTE embedding attention_mask past_key_values alreadycontainsstatus
 outputs = llm(
 inputs_embeds=inputs_embeds_extra,
 attention_mask=extra_mask, # embedding mask
 past_key_values=past,
 use_cache=use_cache,
 output_hidden_states=True,
 return_dict=True,
 **kwargs,
 )
 past = outputs.past_key_values
 # update attn_mask contains embedding forgenerate
 attn_mask = torch.cat([attn_mask, extra_mask], dim=1)

 # ====== processinputoptional======
 # NOTEif inputs_embeds input_idsskip token detectionalready embedding 
 if not has_input_ids:
 input_spans = [] # skipdetection
 else:
 try:
 input_text = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)

 if hasattr(self, "token_classifier_head") and self.token_classifier_head is not None:
 input_spans = self._detect_mol_entities_with_classifier(input_ids[0], input_text)
 else:
 input_spans = self._find_all_mol_spans(input_text)
 except Exception as e:
 if verbose_logging:
 logger.warning(f"[Generate] ⚠️ Failed to detect input entities: {e}")
 input_spans = []

 if input_spans:
 model_dtype = next(self.llm.parameters()).dtype

 for inner_text, end_char in input_spans:
 cleaned_text = (inner_text or "").strip()
 trailing_punct = ",.;:!?"
 while cleaned_text and cleaned_text[-1] in trailing_punct:
 cleaned_text = cleaned_text[:-1].strip()
 while cleaned_text and cleaned_text[0] in trailing_punct:
 cleaned_text = cleaned_text[1:].strip()

 text_to_check = cleaned_text if cleaned_text else inner_text
 is_smiles = False
 if Chem is not None and text_to_check:
 try:
 mol = Chem.MolFromSmiles(text_to_check)
 if mol is not None:
 canonical_smiles = Chem.MolToSmiles(mol)
 if canonical_smiles and len(text_to_check) >= 5:
 is_smiles = True
 except Exception:
 is_smiles = False

 effective_text = text_to_check if is_smiles else (inner_text or "")
 occ_key = (effective_text, int(end_char))
 if occ_key in processed_occurrences:
 continue
 processed_occurrences.add(occ_key)

 mol_emb = None
 if is_smiles and effective_text:
 cache_key = effective_text
 if cache_key in gen_mol_emb_cache:
 mol_emb = gen_mol_emb_cache[cache_key]
 if verbose_logging:
 logger.info(f"[Generate] ✅ Input entity (cached): '{effective_text}'")
 else:
 try:
 with torch.no_grad():
 gvp_embedding_raw = self.gvp_encoder.forward_from_smiles(effective_text).squeeze(0)
 mol_adapter_dtype = next(self.mol_adapter.parameters()).dtype
 gvp_embedding = gvp_embedding_raw.to(dtype=mol_adapter_dtype) if gvp_embedding_raw.dtype != mol_adapter_dtype else gvp_embedding_raw
 mol_emb = self.mol_adapter(gvp_embedding)
 if mol_emb.dtype != model_dtype:
 mol_emb = mol_emb.to(dtype=model_dtype)
 gen_mol_emb_cache[cache_key] = mol_emb
 if verbose_logging:
 logger.info(f"[Generate] ✅ Input entity (fresh): '{effective_text}' -> GVP -> mol_adapter")
 except Exception as e:
 logger.warning(f"[Generate] ⚠️ Failed to process input SMILES '{effective_text}': {e}")
 mol_emb = None
 elif self.enable_diffusion_fallback:
 try:
 h_ctx = outputs.hidden_states[-1][0, -1, :].detach()
 mol_emb = self._black_box_from_hidden_hctx(h_ctx)
 if mol_emb is not None and verbose_logging:
 logger.info(f"[Generate] ✅ Input entity (diffusion): '{inner_text}'")
 except Exception as e:
 logger.warning(f"[Generate] ⚠️ Diffusion failed on input entity '{inner_text}': {e}")
 mol_emb = None

 if mol_emb is not None:
 if mol_emb.device != dev:
 mol_emb = mol_emb.to(device=dev)
 if mol_emb.dtype != model_dtype:
 mol_emb = mol_emb.to(dtype=model_dtype)

 # insert virtual step
 outputs = llm(
 inputs_embeds=mol_emb.view(1, 1, -1),
 attention_mask=torch.cat([attn_mask, torch.ones(1, 1, device=dev, dtype=attn_mask.dtype)], dim=1),
 past_key_values=past,
 use_cache=use_cache,
 output_hidden_states=True,
 return_dict=True,
 **kwargs,
 )
 past = outputs.past_key_values
 attn_mask = torch.cat([attn_mask, torch.ones(1, 1, device=dev, dtype=attn_mask.dtype)], dim=1)
 processed_inner_texts.add(effective_text)

 if input_text.count("</mol>") > 0:
 processed_pair_count = input_text.count("</mol>")

 # ====== loop ======
 generated_ids: List[int] = []
 if has_input_ids:
 prefix_ids = input_ids[0].tolist()
 else:
 # inputs_embeds get prefix_idsemptylist
 prefix_ids = []

 force_detection_next = False # virtual step nextboundarydetectionloopdefault

 for step in range(int(max_new_tokens)):
 logits = outputs.logits[:, -1, :]

 # repetition penalty
 if repetition_penalty and repetition_penalty != 1.0 and generated_ids:
 uniq = list(set(generated_ids))
 logits[:, uniq] = logits[:, uniq] / float(repetition_penalty)
 recent_window = min(10, len(generated_ids))
 recent_tokens_penalty = generated_ids[-recent_window:]
 recent_uniq = list(set(recent_tokens_penalty))
 if recent_uniq:
 logits[:, recent_uniq] = logits[:, recent_uniq] / (float(repetition_penalty) * 1.2)

 # no_repeat_ngramoptional inputs_embeds skip
 if has_input_ids and no_repeat_ngram_size and no_repeat_ngram_size > 0:
 logits = _block_no_repeat_ngrams(logits, prefix_ids, generated_ids, no_repeat_ngram_size)

 next_token = _apply_sampling(logits)
 next_id = int(next_token.item())

 # ===== Asample <mol> token -> i.e.insert virtual step=====
 # NOTE inputs_embeds input_idsskip token detectionalready embedding 
 if not has_input_ids:
 # inputs_embeds skip <mol> token detectionprocess
 pass
 elif next_id == self.mol_token_id:
 current_context_ids = torch.cat(
 [input_ids, torch.tensor([generated_ids], device=dev, dtype=input_ids.dtype)],
 dim=1
 )
 llm_context_text = self.tokenizer.decode(current_context_ids[0].tolist(), skip_special_tokens=False)

 mol_embedding = None
 gnn_path = None
 inner = self._extract_last_between_mol_tags(llm_context_text or "")

 is_smiles = False
 if inner and Chem is not None:
 try:
 is_smiles = (Chem.MolFromSmiles(inner) is not None)
 except Exception:
 is_smiles = False

 if inner and is_smiles:
 try:
 if inner in gen_mol_emb_cache:
 mol_embedding = gen_mol_emb_cache[inner]
 gnn_path = "GNN (cached via <mol>)"
 else:
 model_dtype = next(self.llm.parameters()).dtype
 with torch.no_grad():
 gvp_embedding_raw = self.gvp_encoder.forward_from_smiles(inner).squeeze(0)
 mol_adapter_dtype = next(self.mol_adapter.parameters()).dtype
 gvp_embedding = gvp_embedding_raw.to(dtype=mol_adapter_dtype) if gvp_embedding_raw.dtype != mol_adapter_dtype else gvp_embedding_raw
 mol_embedding = self.mol_adapter(gvp_embedding)
 if mol_embedding.dtype != model_dtype:
 mol_embedding = mol_embedding.to(dtype=model_dtype)
 gen_mol_emb_cache[inner] = mol_embedding
 gnn_path = "GNN (fresh via <mol>)"
 except Exception as e:
 logger.warning(f"[Generate] ⚠️ Failed to process SMILES '{inner}' via <mol>: {e}")
 mol_embedding = None
 elif self.enable_diffusion_fallback:
 try:
 h_ctx_step = outputs.hidden_states[-1][0, -1, :].detach()
 mol_embedding = self._black_box_from_hidden_hctx(h_ctx_step)
 if mol_embedding is not None:
 gnn_path = "Diffusion fallback (via <mol>)"
 except Exception as e:
 logger.warning(f"[Generate] ⚠️ Diffusion fallback failed via <mol>: {e}")
 mol_embedding = None

 if mol_embedding is None:
 # forbid <mol>sample
 logits_block = logits.clone()
 logits_block[0, self.mol_token_id] = float("-inf")
 next_token = _apply_sampling(logits_block)
 next_id = int(next_token.item())
 else:
 model_dtype = next(self.llm.parameters()).dtype
 if mol_embedding.dtype != model_dtype:
 mol_embedding = mol_embedding.to(dtype=model_dtype)
 if mol_embedding.device != dev:
 mol_embedding = mol_embedding.to(device=dev)

 if verbose_logging:
 logger.info(f"[Generate] 🎯 Inserting virtual step via {gnn_path}")

 outputs = llm(
 inputs_embeds=mol_embedding.view(1, 1, -1),
 attention_mask=torch.cat([attn_mask, torch.ones(1, 1, device=dev, dtype=attn_mask.dtype)], dim=1),
 past_key_values=past,
 use_cache=use_cache,
 output_hidden_states=True,
 return_dict=True,
 **kwargs,
 )
 past = outputs.past_key_values
 attn_mask = torch.cat([attn_mask, torch.ones(1, 1, device=dev, dtype=attn_mask.dtype)], dim=1)
 # <mol> generated_ids
 continue

 # ===== generate token =====
 step_ids = next_token # [1,1]
 attn_mask = torch.cat([attn_mask, torch.ones(1, 1, device=dev, dtype=attn_mask.dtype)], dim=1)
 outputs = llm(
 input_ids=step_ids,
 attention_mask=attn_mask,
 past_key_values=past,
 use_cache=use_cache,
 output_hidden_states=True,
 return_dict=True,
 **kwargs,
 )
 past = outputs.past_key_values
 generated_ids.append(next_id)

 # ✅ stop token break
 if stop_on_eos and (next_id in stop_token_ids):
 if verbose_logging:
 tok_txt = ""
 try:
 tok_txt = self.tokenizer.decode([next_id], skip_special_tokens=False)
 except Exception:
 pass
 logger.info(f"[Generate] 🛑 Stop token hit: id={next_id} text={tok_txt!r}")
 break

 # ===== Bboundarydetection virtual step✅ ""=====
 # NOTE inputs_embeds input_idsskip token detectionalready embedding 
 if not has_input_ids:
 # inputs_embeds skipboundarydetection
 continue
 
 try:
 # whetherneedsdetectionboundarytoken/
 should_detect_at_boundary = False
 if force_detection_next:
 should_detect_at_boundary = True
 force_detection_next = False
 elif _is_boundary_token(self.tokenizer, next_id):
 self._boundary_token_count += 1
 if self._boundary_token_count >= int(self._detection_interval):
 should_detect_at_boundary = True
 self._boundary_token_count = 0

 if not should_detect_at_boundary:
 continue

 # detection + overlap
 WINDOW_TOKENS = 2048 # can3072/4096GPU memory/
 OVERLAP_TOKENS = 512 # SMILES/

 current_context_ids = torch.cat([input_ids, torch.tensor([generated_ids], device=dev, dtype=input_ids.dtype)], dim=1)
 seq = current_context_ids[0]
 L = seq.numel()

 start = max(0, L - WINDOW_TOKENS)
 tokens_to_detect = seq[start:] # 
 text_to_detect = self.tokenizer.decode(tokens_to_detect.tolist(), skip_special_tokens=False)

 # compute offsetlengthfor end_char mappingglobal
 # offset computecandecode prefix
 prefix_text = self.tokenizer.decode(seq[:start].tolist(), skip_special_tokens=False) if start > 0 else ""
 input_offset_chars = len(prefix_text)

 detected_spans = self._detect_mol_entities_with_classifier(tokens_to_detect, text_to_detect)
 # mappingglobalif end_char global
 detected_spans = [(t, p + input_offset_chars) for (t, p) in detected_spans]


 current_context_ids = torch.cat(
 [input_ids, torch.tensor([generated_ids], device=dev, dtype=input_ids.dtype)],
 dim=1
 )
 llm_context_text = self.tokenizer.decode(current_context_ids[0].tolist(), skip_special_tokens=False)

 full_text_mol_count = llm_context_text.count("</mol>")

 # detectioncancelconsistent
 text_to_detect = llm_context_text
 tokens_to_detect = current_context_ids[0]
 input_offset_chars = 0

 spans: List[Tuple[str, int]] = []

 # ifnew </mol>rundetection
 if full_text_mol_count > processed_pair_count:
 if hasattr(self, "token_classifier_head") and self.token_classifier_head is not None:
 detected_spans = self._detect_mol_entities_with_classifier(tokens_to_detect, text_to_detect)
 if input_offset_chars > 0:
 detected_spans = [(t, p + input_offset_chars) for t, p in detected_spans]
 spans.extend(detected_spans)

 # boundarydetection
 detected_spans = self._detect_mol_entities_with_classifier(tokens_to_detect, text_to_detect)
 if input_offset_chars > 0:
 detected_spans = [(t, p + input_offset_chars) for t, p in detected_spans]

 # filtergenerate
 input_text_only = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
 generated_text_only = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

 filtered_spans = []
 input_text_len = len(input_text_only)
 for inner_text, end_char_pos in detected_spans:
 if inner_text in generated_text_only:
 # if input_text end_char_pos input_text_len 
 if inner_text not in input_text_only:
 filtered_spans.append((inner_text, end_char_pos))
 else:
 if int(end_char_pos) > int(input_text_len):
 filtered_spans.append((inner_text, end_char_pos))

 # looks_like_molecule 
 for inner_text, end_char_pos in filtered_spans:
 inner = (inner_text or "").strip()
 if _looks_like_molecule(inner):
 spans.append((inner_text, end_char_pos))

 # spans
 uniq = []
 seen_local = set()
 for t, p in spans:
 key = (t, int(p))
 if key in seen_local:
 continue
 seen_local.add(key)
 uniq.append((t, int(p)))
 spans = uniq

 if spans:
 inserted_virtual_steps = False
 model_dtype = next(self.llm.parameters()).dtype

 for inner_text, end_char_pos in spans:
 #
 cleaned_text = (inner_text or "").strip()
 trailing_punct = ",.;:!?"
 while cleaned_text and cleaned_text[-1] in trailing_punct:
 cleaned_text = cleaned_text[:-1].strip()
 while cleaned_text and cleaned_text[0] in trailing_punct:
 cleaned_text = cleaned_text[1:].strip()

 text_to_check = cleaned_text if cleaned_text else (inner_text or "")
 is_smiles = False
 if Chem is not None and text_to_check:
 try:
 mol = Chem.MolFromSmiles(text_to_check)
 if mol is not None:
 canonical_smiles = Chem.MolToSmiles(mol)
 if canonical_smiles and len(text_to_check) >= 5:
 is_smiles = True
 except Exception:
 is_smiles = False

 effective_text = text_to_check if is_smiles else (inner_text or "")

 # ✅ key —— 
 occ_key = (effective_text, int(end_char_pos))
 if occ_key in processed_occurrences:
 continue
 processed_occurrences.add(occ_key)

 mol_emb = None
 if is_smiles and effective_text:
 cache_key = effective_text
 if cache_key in gen_mol_emb_cache:
 mol_emb = gen_mol_emb_cache[cache_key]
 if verbose_logging:
 logger.info(f"[Generate] ✅ Reuse cached embedding for '{effective_text}'")
 else:
 try:
 with torch.no_grad():
 gvp_embedding_raw = self.gvp_encoder.forward_from_smiles(effective_text).squeeze(0)
 mol_adapter_dtype = next(self.mol_adapter.parameters()).dtype
 gvp_embedding = gvp_embedding_raw.to(dtype=mol_adapter_dtype) if gvp_embedding_raw.dtype != mol_adapter_dtype else gvp_embedding_raw
 mol_emb = self.mol_adapter(gvp_embedding)
 if mol_emb.dtype != model_dtype:
 mol_emb = mol_emb.to(dtype=model_dtype)
 gen_mol_emb_cache[cache_key] = mol_emb
 if verbose_logging:
 logger.info(f"[Generate] ✅ Fresh embedding for '{effective_text}' (GVP+adapter)")
 except Exception as e:
 logger.warning(f"[Generate] ⚠️ Failed to process SMILES '{effective_text}': {e}")
 mol_emb = None
 elif self.enable_diffusion_fallback:
 try:
 h_ctx_step2 = outputs.hidden_states[-1][0, -1, :].detach()
 mol_emb = self._black_box_from_hidden_hctx(h_ctx_step2)
 if mol_emb is not None and verbose_logging:
 logger.info(f"[Generate] ✅ Diffusion fallback embedding for '{inner_text}'")
 except Exception as e:
 logger.warning(f"[Generate] ⚠️ Diffusion fallback failed for '{inner_text}': {e}")
 mol_emb = None

 if mol_emb is None:
 continue

 if mol_emb.device != dev:
 mol_emb = mol_emb.to(device=dev)
 if mol_emb.dtype != model_dtype:
 mol_emb = mol_emb.to(dtype=model_dtype)

 # insert virtual step
 outputs = llm(
 inputs_embeds=mol_emb.view(1, 1, -1),
 attention_mask=torch.cat([attn_mask, torch.ones(1, 1, device=dev, dtype=attn_mask.dtype)], dim=1),
 past_key_values=past,
 use_cache=use_cache,
 output_hidden_states=True,
 return_dict=True,
 **kwargs,
 )
 past = outputs.past_key_values
 attn_mask = torch.cat([attn_mask, torch.ones(1, 1, device=dev, dtype=attn_mask.dtype)], dim=1)
 inserted_virtual_steps = True
 processed_inner_texts.add(effective_text)

 processed_pair_count = max(processed_pair_count, full_text_mol_count)

 # insertcontinuesamplecurrent token detection
 if inserted_virtual_steps:
 continue

 except Exception as e:
 logger.warning(f"[Generate] ⚠️ Exception in entity detection/GNN logic: {e}", exc_info=False)

 if not generated_ids:
 # ifgenerate tokenreturnsoriginalinput
 if has_input_ids:
 return input_ids
 else:
 # inputs_embeds returnsoriginalinput input_ids
 # returnsempty tensor
 return torch.empty(1, 0, dtype=torch.long, device=dev)
 
 gen = torch.tensor([generated_ids], device=dev, dtype=torch.long)
 # returnsresultif input_idsreturnsotherwisereturnsgenerate token IDs
 if has_input_ids:
 return torch.cat([input_ids, gen], dim=1)
 else:
 return gen


 # --------------------------- HF save/load ---------------------------
 def state_dict(self, *args, **kwargs):
 # savemodelweightcontainsdefinemodule + llm parameter
 sd = super().state_dict(*args, **kwargs)
 # same storageshared tensor 
 seen = {}
 for k, v in list(sd.items()):
 if not isinstance(v, torch.Tensor):
 continue
 sid = self._storage_id(v)
 if sid in seen:
 sd[k] = v.clone()
 else:
 seen[sid] = k
 return sd

 def save_pretrained(self, save_directory: str, **kwargs):
 """
 - call LLM save_pretrainedsaveweightconfig 
 - savemodeldefinemodule.pt
 - write metadata.json recordfile from_pretrained restore
 """
 os.makedirs(save_directory, exist_ok=True)
 # 1) save LLM
 out = self.llm.save_pretrained(save_directory, **kwargs)

 # 2) savedefinemodule
 extras = {}
 if hasattr(self, "gvp_encoder") and self.gvp_encoder is not None:
 torch.save(self.gvp_encoder.state_dict(), os.path.join(save_directory, "gvp_encoder.pt"))
 extras["gvp_encoder"] = "gvp_encoder.pt"
 if hasattr(self, "mol_adapter") and self.mol_adapter is not None:
 torch.save(self.mol_adapter.state_dict(), os.path.join(save_directory, "mol_adapter.pt"))
 extras["mol_adapter"] = "mol_adapter.pt"
 # NOTEdiffusion_adapter removeLDMoluseLLMhidden states
 # save diffusion_adapter

 # diffusion optionalsaveifneeds
 # if hasattr(self, "diffusion") and self.diffusion is not None:
 # torch.save(self.diffusion.state_dict(), os.path.join(save_directory, "diffusion.pt"))
 # extras["diffusion"] = "diffusion.pt"

 meta = {
 "class": "MolAwareCausalLM",
 "version": 1,
 "extras": extras,
 "mol_token": self.mol_token,
 }
 with open(os.path.join(save_directory, "molaware_metadata.json"), "w", encoding="utf-8") as f:
 json.dump(meta, f, ensure_ascii=False, indent=2)

 return out

 @classmethod
 def from_pretrained(cls, save_directory: str, tokenizer=None,
 diffusion_config=None, diffusion_adapter_config=None,
 layer2_config=None, use_layer2=False,
 **kwargs):
 root = save_directory
 meta_path = os.path.join(root, "molaware_metadata.json")
 has_meta = os.path.isfile(meta_path)

 # 1) parse metadata
 meta = {}
 extras_map = {}
 if has_meta:
 with open(meta_path, "r", encoding="utf-8") as f:
 meta = json.load(f)
 extras_map = meta.get("extras", {}) or {}

 # 2) decide LLM directory <root>/llm <root>
 llm_dir = os.path.join(root, "llm")
 if not has_hf_model_files(llm_dir):
 llm_dir = root
 print(f"[from_pretrained] using llm_dir={llm_dir}")

 # 3) load LLM
 # process torch versionlimitif torch.load limituse safetensors check
 try:
 base_llm = AutoModelForCausalLM.from_pretrained(llm_dir, **kwargs)
 except (ValueError, RuntimeError) as e:
 error_str = str(e)
 if ("torch.load" in error_str and "v2.6" in error_str) or ("CVE-2025-32434" in error_str):
 # torch versionlimituseenvironmentvariablecheck
 # NOTEneeds transformers supports TRANSFORMERS_SAFE_LOADING_DISABLED environmentvariable
 old_val = os.environ.get("TRANSFORMERS_SAFE_LOADING_DISABLED", None)
 try:
 # setenvironmentvariablecheck
 os.environ["TRANSFORMERS_SAFE_LOADING_DISABLED"] = "1"
 # load
 base_llm = AutoModelForCausalLM.from_pretrained(llm_dir, **kwargs)
 except Exception as e2:
 # iffailuse safetensorsif
 import glob
 safetensors_files = glob.glob(os.path.join(llm_dir, "*.safetensors"))
 if safetensors_files:
 try:
 # load safetensors
 base_llm = AutoModelForCausalLM.from_pretrained(
 llm_dir, 
 use_safetensors=True,
 **kwargs
 )
 except Exception as e3:
 raise RuntimeError(
 f"loadmodeltorch versionlimit torch >= 2.6 use safetensors formatmodel"
 f"originalerror: {e}\nfail: {e2}\n safetensors fail: {e3}"
 ) from e3
 else:
 raise RuntimeError(
 f"loadmodeltorch versionlimit torch >= 2.6 use safetensors formatmodel"
 f"originalerror: {e}\nfail: {e2}"
 ) from e2
 finally:
 # restoreenvironmentvariable
 if old_val is not None:
 os.environ["TRANSFORMERS_SAFE_LOADING_DISABLED"] = old_val
 elif "TRANSFORMERS_SAFE_LOADING_DISABLED" in os.environ:
 del os.environ["TRANSFORMERS_SAFE_LOADING_DISABLED"]
 else:
 raise

 # 4) tokenizerdirectory tokenizer save
 if tokenizer is None:
 tokenizer = AutoTokenizer.from_pretrained(root, use_fast=True)

 # 5) construct
 model = cls(llm=base_llm, tokenizer=tokenizer,
 diffusion_config=diffusion_config,
 diffusion_adapter_config=diffusion_adapter_config,
 layer2_config=layer2_config,
 use_layer2=use_layer2)

 # 6) load extras metadata path
 def _maybe_load_sub(sd_path, module_attr):
 if not sd_path:
 return
 path = os.path.join(root, sd_path) if not os.path.isabs(sd_path) else sd_path
 if os.path.isfile(path):
 sd = torch.load(path, map_location="cpu")
 mod = getattr(model, module_attr, None)
 if mod is not None and hasattr(mod, "load_state_dict"):
 # state_dictkeys orprefixuse strict=False 
 mod.load_state_dict(sd, strict=False)

 if has_meta:
 _maybe_load_sub(extras_map.get("gvp_encoder"), "gvp_encoder")
 _maybe_load_sub(extras_map.get("mol_adapter"), "mol_adapter")
 # NOTEdiffusion_adapter old diffusion remove
 # loadcomponent

 return model


 # --------------------------- ---------------------------
 def gradient_checkpointing_enable(self, *args, **kwargs):
 if self.config is not None:
 try:
 self.config.use_cache = False
 except Exception:
 pass
 if hasattr(self.llm, "gradient_checkpointing_enable"):
 try:
 return self.llm.gradient_checkpointing_enable(*args, **kwargs)
 except TypeError:
 return self.llm.gradient_checkpointing_enable()
 return None

 def gradient_checkpointing_disable(self):
 if hasattr(self.llm, "gradient_checkpointing_disable"):
 try:
 out = self.llm.gradient_checkpointing_disable()
 except TypeError:
 out = None
 else:
 out = None
 if self.config is not None:
 try:
 self.config.use_cache = True
 except Exception:
 pass
 return out

 @staticmethod
 def _storage_id(t: torch.Tensor):
 try:
 return t.untyped_storage().data_ptr()
 except Exception:
 return t.storage().data_ptr()
