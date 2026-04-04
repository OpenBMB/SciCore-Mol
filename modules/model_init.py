"""
modelinitializemodule (Optimized)
processmodeltokenizerGNNinitialize
"""
import os
import json
import gc
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from collections import OrderedDict

# latencyloopdependency
from .mol_aware_lm import MolAwareCausalLM


def clean_state_dict(state_dict: Dict[str, Any]) -> OrderedDict:
 """functionremove DDP 'module.' prefixreturns OrderedDict"""
 new_state_dict = OrderedDict()
 for k, v in state_dict.items():
 name = k[7:] if k.startswith("module.") else k
 new_state_dict[name] = v
 return new_state_dict


def init_tokenizer(llm_name: str, mol_token: str = "<mol>") -> AutoTokenizer:
 """
 initializetokenizer
 optimizationset padding_side='right' SFT training
 """
 tokenizer = AutoTokenizer.from_pretrained(llm_name, use_fast=True)
 
 # 1. set pad_token (fix SFTTrainer )
 if tokenizer.pad_token is None:
 tokenizer.pad_token = tokenizer.eos_token
 tokenizer.pad_token_id = tokenizer.eos_token_id
 
 # 2. set padding_side (generatetask)
 tokenizer.padding_side = "right"
 
 # 3. token
 to_add = []
 current_vocab = tokenizer.get_vocab()
 if mol_token not in current_vocab:
 to_add.append(mol_token)
 
 # Llama model Llama 3 token
 # Qwen / Mistral model token chat 
 llm_name_lower = llm_name.lower()
 if "llama" in llm_name_lower:
 special_tokens = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
 for t in special_tokens:
 if t not in current_vocab:
 to_add.append(t)
 
 if to_add:
 tokenizer.add_special_tokens({"additional_special_tokens": to_add})
 
 return tokenizer


def init_llm(llm_name: str, tokenizer: AutoTokenizer, bf16: bool = True, device: str = "cuda:0") -> AutoModelForCausalLM:
 """
 initializeLLM
 
 ifpath checkpoint pathcontains pytorch_model.bin model.safetensors
 llm/ directorycall split_llm_extras.py 
 """
 import os
 from pathlib import Path
 
 # check torch version
 torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
 requires_torch_26 = torch_version < (2, 6)
 
 # checkmodeldirectorywhether safetensors file
 model_path = Path(llm_name)
 
 # ifpathcheckwhether checkpoint pathneeds
 if not model_path.exists():
 # checkwhether checkpoint/llm pathdirectorycontainsweight
 parent_path = model_path.parent
 if parent_path.exists():
 bin_path = parent_path / "pytorch_model.bin"
 safetensors_path = parent_path / "model.safetensors"
 if bin_path.exists() or safetensors_path.exists():
 print(f"📦 detection checkpoint path llm directory: {llm_name}")
 print(f" directory: {parent_path}")
 try:
 split_script_path = Path(__file__).parent.parent / "scripts" / "ckpt" / "split_llm_extras.py"
 if split_script_path.exists():
 import importlib.util
 spec = importlib.util.spec_from_file_location("split_llm_extras", split_script_path)
 split_module = importlib.util.module_from_spec(spec)
 spec.loader.exec_module(split_module)
 success = split_module.split_checkpoint(str(parent_path), str(parent_path))
 if success:
 print(f"✅ Checkpoint completecheckpath: {llm_name}")
 model_path = Path(llm_name) # value
 except Exception as e:
 print(f"⚠️ fail: {e}")
 has_safetensors = False
 has_bin_files = False
 if model_path.exists():
 safetensors_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("model*.safetensors"))
 has_safetensors = len(safetensors_files) > 0
 bin_files = list(model_path.glob("*.bin")) + list(model_path.glob("pytorch_model*.bin"))
 has_bin_files = len(bin_files) > 0
 
 # if torch < 2.6 .bin file CPU convert safetensorscheck
 if requires_torch_26 and not has_safetensors and has_bin_files:
 import warnings
 warnings.warn(
 f"⚠️ Torch version {torch.__version__} < 2.6, model has only .bin files. "
 f"Trying CPU-side auto-conversion to safetensors to bypass the security check."
 )
 try:
 print(f"[Model Init] Converting {llm_name} to safetensors (CPU, dtype={'bf16' if bf16 else 'fp32'})...")
 from transformers import AutoConfig
 # transformers checkallowload .bin
 os.environ["TRANSFORMERS_SAFE_LOADING_DISABLED"] = "1"
 # choose bin file indexprompt/hintprocess
 index_files = list(model_path.glob("pytorch_model*.bin.index.json"))
 if index_files:
 raise RuntimeError(
 "Model appears to be sharded (.bin.index.json found); please convert manually "
 "or upgrade torch>=2.6 to load sharded .bin safely."
 )
 # first bin file
 bin_file = sorted(list(model_path.glob("*.bin")) + list(model_path.glob("pytorch_model*.bin")))[0]
 print(f"[Model Init] Loading bin weights from: {bin_file.name}")
 state = torch.load(bin_file, map_location="cpu")
 # state_dict 
 if isinstance(state, dict) and "state_dict" in state:
 state = state["state_dict"]
 # load config buildmodel
 config = AutoConfig.from_pretrained(llm_name)
 temp_model = AutoModelForCausalLM.from_config(config)
 missing, unexpected = temp_model.load_state_dict(state, strict=False)
 if missing or unexpected:
 print(f"[Model Init] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
 # save safetensors
 temp_model.save_pretrained(
 llm_name,
 safe_serialization=True,
 max_shard_size="5GB"
 )
 del temp_model
 if torch.cuda.is_available():
 torch.cuda.empty_cache()
 print(f"[Model Init] ✅ Conversion completed. Safetensors files saved.")
 has_safetensors = True
 # restoredefaultset
 os.environ.pop("TRANSFORMERS_SAFE_LOADING_DISABLED", None)
 except Exception as conv_e:
 import traceback
 traceback.print_exc()
 raise RuntimeError(
 f"Failed to auto-convert model to safetensors: {conv_e}\n"
 f"Please manually convert or upgrade torch to >= 2.6"
 ) from conv_e
 elif requires_torch_26 and not has_safetensors:
 import warnings
 warnings.warn(
 f"⚠️ Torch version {torch.__version__} < 2.6, and model has no safetensors files. "
 f"Transformers requires torch>=2.6 to load .bin files due to security (CVE-2025-32434).\n"
 f"Solutions:\n"
 f" 1. Upgrade torch: pip install torch>=2.6\n"
 f" 2. Convert model to safetensors format\n"
 f" 3. Downgrade transformers to a version before this check"
 )
 
 # use low_cpu_mem_usage=True loadmemory
 # safetensorsif
 try:
 llm = AutoModelForCausalLM.from_pretrained(
 llm_name,
 dtype=torch.bfloat16 if bf16 else torch.float32, # use dtype torch_dtype
 low_cpu_mem_usage=True,
 use_safetensors=True if has_safetensors else None, # if safetensors use
 device_map=None, # to(device)
 trust_remote_code=True
 ).to(device)
 except Exception as e:
 # if safetensors loadfail .binneeds torch >= 2.6
 if "safetensors" in str(e).lower() or "use_safetensors" in str(e).lower():
 if requires_torch_26:
 raise RuntimeError(
 f"Failed to load model: {e}\n"
 f"Your torch version ({torch.__version__}) is < 2.6, which is required to load .bin files.\n"
 f"Please upgrade torch: pip install 'torch>=2.6'"
 ) from e
 # torch >= 2.6can .bin
 llm = AutoModelForCausalLM.from_pretrained(
 llm_name,
 dtype=torch.bfloat16 if bf16 else torch.float32,
 low_cpu_mem_usage=True,
 use_safetensors=False, # use .bin
 device_map=None,
 trust_remote_code=True
 ).to(device)
 else:
 raise
 
 # vocab size
 old_vocab_size = llm.get_input_embeddings().weight.shape[0]
 new_vocab_size = len(tokenizer)
 
 if old_vocab_size != new_vocab_size:
 # rank 0 print
 import torch.distributed as dist
 if not dist.is_initialized() or dist.get_rank() == 0:
 print(f"[Model Init] Resizing token embeddings: {old_vocab_size} -> {new_vocab_size}")
 
 # mean_resizing=True use embedding valueinitialize tokenrandominitialize
 llm.resize_token_embeddings(new_vocab_size, mean_resizing=True)
 
 # syncconfig
 llm.config.vocab_size = len(tokenizer)
 llm.config.pad_token_id = tokenizer.pad_token_id
 llm.config.eos_token_id = tokenizer.eos_token_id
 llm.config.bos_token_id = tokenizer.bos_token_id
 
 return llm


def init_model(
 cfg: Dict[str, Any],
 tokenizer: AutoTokenizer,
 llm: AutoModelForCausalLM,
 device: str = "cuda:0",
) -> MolAwareCausalLM:
 """initializeMolAwareCausalLMmodel"""
 mol_token = cfg.get("tokens", {}).get("mol_token", "<mol>")
 train_conf = cfg.get("train", {}) or {}

 # ===== Diffusion =====
 # if cfg["train"]["use_diffusion"] False diffusion
 # initialize diffusion/diffusion_adapter moduleGPU memory
 use_diffusion = train_conf.get("use_diffusion", True)

 if use_diffusion:
 diffusion_conf = cfg.get("diffusion", {}) or {}
 diff_conf = diffusion_conf.get("diffusion", {}) or {}
 diff_adp_conf = diffusion_conf.get("adapter", {}) or {}
 else:
 diffusion_conf = {}
 diff_conf = {}
 diff_adp_conf = {}
 
 # --- optimizationdevice ---
 # allowvia env config diffusion modelGPU memory
 if diff_conf:
 diffusion_device = diff_conf.get("device")
 if not diffusion_device or diffusion_device == "cuda:0":
 env_device = os.environ.get("DIFFUSION_DEVICE")
 if env_device:
 diffusion_device = env_device
 elif device.startswith("cuda:"):
 #
 try:
 curr_id = int(device.split(":")[-1])
 if torch.cuda.device_count() > curr_id + 1:
 diffusion_device = f"cuda:{curr_id + 1}"
 else:
 diffusion_device = device
 except Exception:
 diffusion_device = device
 else:
 diffusion_device = device
 diff_conf["device"] = diffusion_device
 if diffusion_device != device:
 print(f"📌 Diffusion model placed on {diffusion_device} (Main LLM on {device})")

 # checkwhether GNN use_offline_spans=False GNN path
 use_offline_spans = cfg.get("train", {}).get("use_offline_spans", False)
 disable_gnn = not use_offline_spans # if use_offline_spans=False GNN
 
 # checkwhetheruse LDMolstage1 SFT trainingGPU memory
 # if use_diffusion=Falsedefault LDMol LDMol dependency diffusion
 use_ldmol = train_conf.get("use_ldmol", use_diffusion)
 # whetherskip LDMol text_encoderinference Qwen GPU memory
 ldmol_skip_text_encoder = train_conf.get("ldmol_skip_text_encoder", False)
 
 if not use_ldmol:
 print("📌 LDMol disabled (use_ldmol=False)")
 elif ldmol_skip_text_encoder:
 print("📌 LDMol text_encoder skipped (ldmol_skip_text_encoder=True, will reuse main Qwen)")
 
 # checkwhetheruse Layer2forreactionyieldprediction
 use_layer2 = train_conf.get("use_layer2", False)
 layer2_config = cfg.get("layer2", {}) or {}
 
 if use_layer2:
 print("📌 Layer2 enabled (use_layer2=True)")
 else:
 print("📌 Layer2 disabled (use_layer2=False)")
 
 # initializemodel
 model = MolAwareCausalLM(
 llm=llm,
 tokenizer=tokenizer,
 mol_token=mol_token,
 proxy=cfg.get("network", {}).get("proxy"),
 debug=False,
 diffusion_config=diff_conf,
 diffusion_adapter_config=diff_adp_conf,
 disable_gnn=disable_gnn, # GNN 
 use_ldmol=use_ldmol, # whetheruse LDMol
 ldmol_skip_text_encoder=ldmol_skip_text_encoder, # whetherskip LDMol text_encoder
 layer2_config=layer2_config, # Layer2 config
 use_layer2=use_layer2, # whetheruse Layer2
 ).to(device)
 
 # --- weightloadoptimization ---
 checkpoint_dir = cfg.get("paths", {}).get("checkpoint_dir")
 
 # 1. Checkpoint directoryload
 if checkpoint_dir:
 checkpoint_path = Path(checkpoint_dir)
 if checkpoint_path.exists():
 print(f"📂 Loading weights from checkpoint: {checkpoint_dir}")
 load_model_weights_from_checkpoint_dir(model, checkpoint_dir, device)
 else:
 print(f"⚠️ Checkpoint directory: {checkpoint_dir}")
 print(f" skip checkpoint loadusedefaultinitialize")
 
 # 2. otherwisepathload (Legacy / Fine-grained control)
 else:
 # GNN
 gnn_path = cfg.get("paths", {}).get("gnn_state_dict_path")
 if gnn_path and os.path.exists(gnn_path):
 try:
 sd = torch.load(gnn_path, map_location="cpu")
 sd = sd.get("model_state_dict", sd)
 model.gvp_encoder.load_state_dict(clean_state_dict(sd), strict=False)
 print(f"✅ Loaded GVPEncoder from {gnn_path}")
 except Exception as e:
 print(f"⚠️ Load GVP failed: {e}")
 
 # adapter
 load_additional_weights(model, cfg, device)
 
 # initialize GNN taskhead (ifneeds)
 use_gnn_tasks = cfg.get("train", {}).get("use_gnn_tasks", False)
 if use_gnn_tasks or (checkpoint_dir and os.path.exists(checkpoint_dir)):
 init_gnn_task_heads(model, cfg, device)
 
 # freeze/frozenstrategy
 apply_freeze_config(model, cfg)
 
 # GPU memory
 torch.cuda.empty_cache()
 
 return model


def init_gnn_task_heads(model: MolAwareCausalLM, cfg: Dict[str, Any], device: str):
 """initializeGNNtaskhead"""
 if not hasattr(model, "gvp_encoder") or model.gvp_encoder is None:
 return
 
 try:
 train_cfg = cfg.get("train", {})
 model.gvp_encoder.init_task_heads(
 num_reg_tasks=train_cfg.get("gnn_num_reg_tasks", 5),
 num_cls_tasks=train_cfg.get("gnn_num_cls_tasks", 1),
 head_hidden_dim=train_cfg.get("gnn_head_hidden_dim", None),
 head_dropout=float(train_cfg.get("gnn_head_dropout", 0.1)),
 )
 # initializeheadcorrectdevice
 model.gvp_encoder.to(device) 
 print(f"✅ GNN Task Heads initialized.")
 except Exception as e:
 print(f"⚠️ GVP head init failed: {e}")


def load_model_weights_from_checkpoint_dir(model: MolAwareCausalLM, ckpt_dir: str, device: str):
 """
 checkpointdirectoryloadweight
 optimizationoptimizationmodelloadmemory
 
 if checkpoint directory llm/ directorycall split_llm_extras.py 
 """
 ckpt_dir = Path(ckpt_dir)
 
 if not ckpt_dir.exists():
 print(f"❌ Checkpoint directory: {ckpt_dir}")
 return
 
 # checkwhetherneeds checkpoint
 llm_dir = ckpt_dir / "llm"
 
 # if llm directory checkpoint directory pytorch_model.bin model.safetensorsneeds
 if not llm_dir.exists():
 bin_path = ckpt_dir / "pytorch_model.bin"
 safetensors_path = ckpt_dir / "model.safetensors"
 if bin_path.exists() or safetensors_path.exists():
 print(f"📦 detection checkpointneeds: {ckpt_dir}")
 print(f" call split_llm_extras.py ...")
 try:
 # function
 split_script_path = Path(__file__).parent.parent / "split_llm_extras.py"
 if split_script_path.exists():
 # dynamic
 import importlib.util
 spec = importlib.util.spec_from_file_location("split_llm_extras", split_script_path)
 split_module = importlib.util.module_from_spec(spec)
 spec.loader.exec_module(split_module)
 
 # callfunction
 success = split_module.split_checkpoint(str(ckpt_dir), str(ckpt_dir))
 if success:
 print(f"✅ Checkpoint complete")
 # check llm_dirshould
 llm_dir = ckpt_dir / "llm"
 else:
 print(f"⚠️ Checkpoint failcontinueload...")
 else:
 print(f"⚠️ split_llm_extras.py: {split_script_path}")
 except Exception as e:
 import traceback
 print(f"⚠️ fail: {e}")
 traceback.print_exc()
 print(f" run: python {split_script_path} check checkpoint path")
 
 # 1. load LLM weight
 if llm_dir.exists():
 try:
 # use from_pretrained loadinternalprocessloadmemory (low_cpu_mem_usage)
 # merge state_dict memory
 print(f"⏳ Loading LLM via from_pretrained (memory efficient)...")
 # loadmodelextract state_dictor model.llm load
 # model.llm call HF load
 # NOTEneeds model.llm HF model
 
 # A: ifloadweight load_state_dict safetensors 
 # B (): use transformers load_sharded_checkpoint 
 from transformers.modeling_utils import load_sharded_checkpoint
 
 # check safetensors bin
 safetensors_index = llm_dir / "model.safetensors.index.json"
 safetensors_file = llm_dir / "model.safetensors"
 is_safetensors = safetensors_index.exists() or safetensors_file.exists()

 if is_safetensors:
 from safetensors.torch import load_file
 if safetensors_file.exists():
 # file
 sd = load_file(str(safetensors_file), device="cpu")
 model.llm.load_state_dict(sd, strict=False)
 else:
 # safetensors (HF supportsprocess)
 # loadprocess
 load_sharded_checkpoint(model.llm, str(llm_dir), strict=False, prefer_safe=True)
 else:
 # PyTorch Bin
 bin_file = llm_dir / "pytorch_model.bin"
 bin_index = llm_dir / "pytorch_model.bin.index.json"
 if bin_file.exists():
 sd = torch.load(str(bin_file), map_location="cpu")
 model.llm.load_state_dict(sd, strict=False)
 elif bin_index.exists():
 load_sharded_checkpoint(model.llm, str(llm_dir), strict=False, prefer_safe=False)
 
 print(f"✅ Loaded LLM weights from {llm_dir}")
 
 except Exception as e:
 print(f"⚠️ Optimized LLM load failed: {e}, falling back to legacy...")
 # Fallback ()
 pass 

 # 2. load Extras (use clean_state_dict)
 extras_dir = ckpt_dir / "extras"
 if extras_dir.exists():
 def _load_component(name, filename):
 path = extras_dir / filename
 comp = getattr(model, name, None)
 if path.exists() and comp is not None:
 try:
 # checkfilesize
 file_size = path.stat().st_size
 if file_size == 0:
 print(f"⚠️ Skipping {name}: checkpoint file {filename} is empty (0 bytes)")
 return
 
 sd = torch.load(path, map_location="cpu")
 if isinstance(sd, dict) and "model_state_dict" in sd: sd = sd["model_state_dict"]
 elif isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
 
 # checkwhetheremptyweight [0]
 empty_keys = []
 for k, v in sd.items():
 if isinstance(v, torch.Tensor) and v.numel() == 0:
 empty_keys.append(k)
 
 if empty_keys:
 print(f"⚠️ Skipping {name}: checkpoint contains {len(empty_keys)} empty weights (shape [0])")
 print(f" example: {empty_keys[:3]}...")
 return
 
 comp.load_state_dict(clean_state_dict(sd), strict=False)
 print(f"✅ Loaded {name} from {filename}")
 except Exception as e:
 print(f"⚠️ Failed to load {name}: {e}")

 _load_component("gvp_encoder", "gvp_encoder.pt")
 _load_component("mol_adapter", "mol_adapter.pt")
 _load_component("diffusion_adapter", "diffusion_adapter.pt")
 
 # 3. CPU memory
 gc.collect()
 torch.cuda.empty_cache()


def load_additional_weights(model: MolAwareCausalLM, cfg: Dict[str, Any], device: str):
 """loadweight (Unified)"""
 paths = cfg.get("paths", {})
 
 def _load_single(path_key, model_attr, label):
 path = paths.get(path_key)
 comp = getattr(model, model_attr, None)
 if path and os.path.exists(path) and comp is not None:
 try:
 # checkfilesize
 file_size = Path(path).stat().st_size
 if file_size == 0:
 print(f"⚠️ Skipping {label}: checkpoint file is empty (0 bytes): {path}")
 return
 
 sd = torch.load(path, map_location="cpu")
 if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
 
 # checkwhetheremptyweight
 empty_keys = [k for k, v in sd.items() if isinstance(v, torch.Tensor) and v.numel() == 0]
 if empty_keys:
 print(f"⚠️ Skipping {label}: checkpoint contains {len(empty_keys)} empty weights (shape [0])")
 print(f" file: {path}")
 return
 
 comp.load_state_dict(clean_state_dict(sd), strict=False)
 print(f"✅ Loaded {label} from {path}")
 except Exception as e:
 print(f"⚠️ Failed to load {label}: {e}")

 _load_single("gnn_mlp_state_dict_path", "mol_adapter", "mol_adapter")
 _load_single("diffusion_adapter_state_dict_path", "diffusion_adapter", "diffusion_adapter")


def apply_freeze_config(model: MolAwareCausalLM, cfg: Dict[str, Any]):
 """freeze/frozenconfig"""
 train_cfg = cfg.get("train", {})
 
 # functionfreeze/frozenmodule
 def _freeze(module, name):
 for p in module.parameters():
 p.requires_grad = False
 print(f"🔒 Frozen {name}")

 if train_cfg.get("freeze_llm", False):
 for n, p in model.llm.named_parameters():
 if 'embed_tokens' not in n: # embedding trainingtoken
 p.requires_grad = False
 print("🔒 Frozen LLM (except embed_tokens)")

 if train_cfg.get("freeze_gnn", False) and getattr(model, "gvp_encoder", None):
 _freeze(model.gvp_encoder, "GVP Encoder")

 if train_cfg.get("freeze_mol_adapter", False) and getattr(model, "mol_adapter", None):
 _freeze(model.mol_adapter, "Mol Adapter")

 if train_cfg.get("freeze_diffusion", True) and getattr(model, "diffusion", None):
 _freeze(model.diffusion, "Diffusion Model")
 
 if train_cfg.get("freeze_diffusion_adapter", True) and getattr(model, "diffusion_adapter", None):
 _freeze(model.diffusion_adapter, "Diffusion Adapter")


def init_offline_token_classifier(
 llm: AutoModelForCausalLM,
 mlp_token_classifier_path: Optional[str],
 device: str = "cuda:0",
) -> Optional[nn.Module]:
 """initializetokenclass"""
 if not mlp_token_classifier_path:
 print(f"⚠️ mlp_token_classifier_path is not set in config")
 return None
 
 if not os.path.exists(mlp_token_classifier_path):
 print(f"⚠️ Token classifier file not found: {mlp_token_classifier_path}")
 return None
 
 try:
 print(f"📂 Loading token classifier from: {mlp_token_classifier_path}")
 # needstrainingclassconsistent
 # if config readdefault
 hidden_size = llm.config.hidden_size
 print(f" Current model hidden_size: {hidden_size}")
 
 # check checkpoint hidden_size
 ckpt_for_check = torch.load(mlp_token_classifier_path, map_location="cpu")
 if isinstance(ckpt_for_check, dict):
 if "state_dict" in ckpt_for_check:
 ckpt_sd = ckpt_for_check["state_dict"]
 elif "model_state_dict" in ckpt_for_check:
 ckpt_sd = ckpt_for_check["model_state_dict"]
 else:
 ckpt_sd = ckpt_for_check
 
 # first Linear layerweightoriginal hidden_size
 for key, value in ckpt_sd.items():
 # remove prefix
 clean_key = key.replace("classifier.", "").replace("token_classifier.", "").replace("module.", "")
 if "weight" in clean_key and len(value.shape) == 2:
 # first Linear layerinputdimension hidden_size
 ckpt_hidden_size = value.shape[1]
 print(f" Checkpoint hidden_size: {ckpt_hidden_size} (from weight shape: {value.shape})")
 if ckpt_hidden_size != hidden_size:
 print(f" ⚠️ WARNING: Hidden size mismatch! Checkpoint was trained with {ckpt_hidden_size}, "
 f"but current model has {hidden_size}. This classifier cannot be used.")
 return None
 break
 
 token_head = nn.Sequential(
 nn.Linear(hidden_size, 128),
 nn.ReLU(),
 nn.Dropout(0.1),
 nn.Linear(128, 2)
 ).to(device)
 
 print(f" Loading checkpoint...")
 # load checkpointload
 ckpt = ckpt_for_check
 
 # check checkpoint 
 if isinstance(ckpt, dict):
 print(f" Checkpoint keys: {list(ckpt.keys())[:10]}...") # 10key
 if "state_dict" in ckpt:
 ckpt = ckpt["state_dict"]
 elif "model_state_dict" in ckpt:
 ckpt = ckpt["model_state_dict"]
 
 # use clean_state_dict remove potential module. prefixfilter key
 clean_sd = clean_state_dict(ckpt)
 print(f" Cleaned state_dict keys (first 10): {list(clean_sd.keys())[:10]}...")
 
 final_sd = OrderedDict()
 for k, v in clean_sd.items():
 # key format
 if k.startswith("classifier."):
 final_sd[k.replace("classifier.", "")] = v
 elif k.startswith("token_classifier."):
 final_sd[k.replace("token_classifier.", "")] = v
 elif not k.startswith("module.") and not "." in k or k.count(".") <= 1:
 # if key classparameterlayeruse
 final_sd[k] = v
 
 print(f" Final state_dict keys: {list(final_sd.keys())}")
 
 if len(final_sd) == 0:
 print(f"⚠️ No matching keys found in checkpoint. Available keys: {list(clean_sd.keys())[:20]}...")
 return None
 
 # loadif strict=False fail strict=False
 try:
 token_head.load_state_dict(final_sd, strict=True)
 print(f" ✅ Loaded with strict=True")
 except Exception as e:
 print(f" ⚠️ Strict loading failed: {e}, trying strict=False...")
 missing_keys, unexpected_keys = token_head.load_state_dict(final_sd, strict=False)
 if missing_keys:
 print(f" ⚠️ Missing keys: {missing_keys[:5]}...")
 if unexpected_keys:
 print(f" ⚠️ Unexpected keys: {unexpected_keys[:5]}...")
 
 token_head.eval()
 
 # freeze/frozen
 for p in token_head.parameters():
 p.requires_grad = False
 
 print(f"✅ Loaded offline token classifier successfully")
 return token_head
 except Exception as e:
 print(f"❌ Failed to load token classifier: {e}")
 import traceback
 traceback.print_exc()
 return None