import torch
import os
import json
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
import sys


def split_checkpoint(checkpoint_path: str, output_dir: str = None):
 """
 checkpoint weight LLM Extras
 
 Args:
 checkpoint_path: checkpoint directorypathcontains pytorch_model.bin
 output_dir: outputdirectorydefault checkpoint_path
 """
 if output_dir is None:
 output_dir = checkpoint_path
 
 checkpoint_path = Path(checkpoint_path)
 output_dir = Path(output_dir)
 
 # checkweightfilesupportsformat
 bin_path = checkpoint_path / "pytorch_model.bin"
 safetensors_path = checkpoint_path / "model.safetensors"
 
 # checkwhether safetensors filemodel-00001-of-00005.safetensors 
 safetensors_index_path = checkpoint_path / "model.safetensors.index.json"
 sharded_safetensors = False
 if safetensors_index_path.exists():
 try:
 with open(safetensors_index_path, 'r') as f:
 index_data = json.load(f)
 if "weight_map" in index_data:
 sharded_safetensors = True
 print(f"🔄 detection safetensors file")
 except:
 pass
 
 # check global_step directoryDeepSpeed ZeRO checkpoint
 global_step_dirs = sorted([d for d in checkpoint_path.iterdir() if d.is_dir() and d.name.startswith("global_step")])
 
 if not bin_path.exists() and not safetensors_path.exists() and not sharded_safetensors and not global_step_dirs:
 print(f"❌ weightfile: {checkpoint_path}")
 print(f" : pytorch_model.bin, model.safetensors, safetensors, global_step* directory")
 return False
 
 # use safetensors
 if sharded_safetensors:
 print(f"🔄 load safetensors weight...")
 from safetensors.torch import load_file
 full_state_dict = {}
 with open(safetensors_index_path, 'r') as f:
 index_data = json.load(f)
 weight_map = index_data.get("weight_map", {})
 # allneedsloadfile
 shard_files = set(weight_map.values())
 for shard_file in sorted(shard_files):
 shard_path = checkpoint_path / shard_file
 if shard_path.exists():
 print(f" load: {shard_file} ...")
 shard_dict = load_file(str(shard_path))
 full_state_dict.update(shard_dict)
 else:
 print(f" ⚠️ warning: file: {shard_file}")
 elif safetensors_path.exists():
 print(f"🔄 loadweight (safetensors): {safetensors_path} ...")
 from safetensors.torch import load_file
 full_state_dict = load_file(str(safetensors_path))
 elif global_step_dirs:
 # new global_step directoryrestore
 latest_global_step = global_step_dirs[-1]
 print(f"🔄 detection DeepSpeed ZeRO checkpoint {latest_global_step.name} restore...")
 
 # modelstatusfile
 model_state_files = list(latest_global_step.glob("*model_states.pt"))
 if not model_state_files:
 print(f"❌ {latest_global_step} model_states.pt file")
 return False
 
 # loadallmodelstatus
 full_state_dict = {}
 for model_state_file in sorted(model_state_files):
 print(f" loadmodelstatus: {model_state_file.name} ...")
 state = torch.load(model_state_file, map_location="cpu")
 # DeepSpeed ZeRO formatstate contains 'module' key
 if isinstance(state, dict):
 if 'module' in state:
 state = state['module']
 elif 'model' in state:
 state = state['model']
 # merge full_state_dict
 if isinstance(state, dict):
 full_state_dict.update(state)
 else:
 print(f" ⚠️ warning: {model_state_file.name} formatskip")
 
 if not full_state_dict:
 print(f"❌ {latest_global_step} loadmodelweight")
 return False
 print(f"✅ success {latest_global_step.name} load {len(full_state_dict)} weight")
 else:
 print(f"🔄 loadweight (bin): {bin_path} ...")
 full_state_dict = torch.load(bin_path, map_location="cpu")
 
 llm_sd = {}
 extras_sd = {}
 
 print("🔄 weight...")
 keys = list(full_state_dict.keys())
 
 # define extras keyforrecognition LLM weight
 extras_keywords = ["gvp_encoder", "mol_adapter", "diffusion_adapter", "diffusion"]
 
 for k in keys:
 # checkwhether extras weight
 is_extras = any(x in k for x in extras_keywords)
 
 if is_extras:
 extras_sd[k] = full_state_dict[k]
 elif k.startswith("llm."):
 # llm. prefixprefix llm_sd
 new_k = k[4:] 
 llm_sd[new_k] = full_state_dict[k]
 else:
 # prefix LLM weight
 # checkwhethercontains LLM layer
 llm_keywords = ["embed", "layers", "norm", "lm_head", "model.", "transformer."]
 if any(x in k for x in llm_keywords):
 llm_sd[k] = full_state_dict[k]
 else:
 # keydefault LLM weight
 llm_sd[k] = full_state_dict[k]
 
 print(f"📊 result: LLMweight={len(llm_sd)} keys, Extrasweight={len(extras_sd)} keys")
 
 # ================= save LLM (fix Config ) =================
 llm_save_dir = output_dir / "llm"
 llm_save_dir.mkdir(parents=True, exist_ok=True)
 
 print(f"💾 save LLM {llm_save_dir} ...")
 try:
 # 1. load Configpriorityllm/directory > checkpointdirectory > directory
 config = None
 llm_config_path = checkpoint_path / "llm" / "config.json"
 root_config_path = checkpoint_path / "config.json"
 parent_config_path = checkpoint_path.parent / "config.json"
 
 if llm_config_path.exists():
 print(f"📂 llm/ directoryload config: {llm_config_path}")
 config = AutoConfig.from_pretrained(str(llm_config_path))
 elif root_config_path.exists():
 print(f"📂 checkpoint directoryload config: {root_config_path}")
 config = AutoConfig.from_pretrained(str(checkpoint_path))
 elif parent_config_path.exists():
 print(f"📂 directoryload config: {parent_config_path}")
 config = AutoConfig.from_pretrained(str(checkpoint_path.parent))
 else:
 print("⚠️ config.json checkpoint directorydetection...")
 try:
 config = AutoConfig.from_pretrained(str(checkpoint_path))
 except Exception as e:
 print(f"⚠️ checkpoint load config: {e}")
 # llm directoryloadi.e.pathAutoConfig 
 try:
 llm_dir = checkpoint_path / "llm"
 if llm_dir.exists():
 config = AutoConfig.from_pretrained(str(llm_dir))
 print(f"✅ llm/ directorysuccessload config")
 except:
 pass
 
 if config is None:
 print("⚠️ load configuseweight vocab_size")
 # create configweightvalue
 # NOTEAutoConfig fileno need to
 # modelinferifpathcontainsmodel
 model_name = str(checkpoint_path)
 if "llama" in model_name.lower():
 config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
 elif "mistral" in model_name.lower():
 config = AutoConfig.from_pretrained("mistralai/Mistral-7B-v0.1")
 else:
 # defaultuse LLaMA
 config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
 print(f"⚠️ usedefault configweightvalue")

 # 2. fixdetectionweightvocabularysize Config
 # embeddinglayersupports key format
 embed_weight = None
 possible_keys = [
 "model.embed_tokens.weight", # LLaMA/Mistral format
 "embed_tokens.weight", # model prefix
 "transformer.wte.weight", # GPT-2 format
 "model.embedding.weight", # format
 ]
 
 # key
 for key in possible_keys:
 if key in llm_sd:
 embed_weight = llm_sd[key]
 print(f"🔍 embeddinglayer: {key}, shape={embed_weight.shape}")
 break
 
 # ifallcontains "embed" key
 if embed_weight is None:
 for k, v in llm_sd.items():
 if "embed" in k.lower() and "weight" in k.lower() and len(v.shape) == 2:
 embed_weight = v
 print(f"🔍 detectionembeddinglayer: {k}, shape={embed_weight.shape}")
 break
 
 if embed_weight is not None:
 real_vocab_size = embed_weight.shape[0]
 if config.vocab_size != real_vocab_size:
 print(f"🔧 detectionvocabularysize: Config({config.vocab_size}) -> Weights({real_vocab_size})")
 print(f"🔧 config.vocab_size = {real_vocab_size}")
 config.vocab_size = real_vocab_size
 else:
 print("⚠️ warning: embeddinglayerweight vocab_size")
 print(f" key example: {list(llm_sd.keys())[:5]}...")
 
 # 3. use Config initializemodelloadweight
 model = AutoModelForCausalLM.from_config(config)
 
 # loadweight (shouldmatch)
 model.load_state_dict(llm_sd, strict=True)
 
 # 4. save HF format (safetensors + config.json)
 model.save_pretrained(str(llm_save_dir), safe_serialization=True)
 
 # save tokenizer (if checkpoint directory tokenizer file)
 try:
 from transformers import AutoTokenizer
 tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
 tokenizer.save_pretrained(str(llm_save_dir))
 print("✅ Tokenizer copysave")
 except Exception as e:
 print(f"⚠️ Tokenizer file: {e}")
 
 print("✅ LLM savesuccess (safetensorsformat)")
 
 except Exception as e:
 import traceback
 print(f"⚠️ LLM save_pretrained fail: {e}")
 traceback.print_exc()
 print("save pytorch_model.bin ...")
 torch.save(llm_sd, str(llm_save_dir / "pytorch_model.bin"))
 
 # ================= save Extras =================
 extras_save_dir = output_dir / "extras"
 extras_save_dir.mkdir(parents=True, exist_ok=True)
 
 gvp_only = {k.replace("gvp_encoder.", ""): v for k, v in extras_sd.items() if "gvp_encoder" in k}
 mol_only = {k.replace("mol_adapter.", ""): v for k, v in extras_sd.items() if "mol_adapter" in k}
 diff_only = {k.replace("diffusion_adapter.", ""): v for k, v in extras_sd.items() if "diffusion_adapter" in k}

 if gvp_only: 
 torch.save(gvp_only, str(extras_save_dir / "gvp_encoder.pt"))
 print(f"✅ save GVP encoder ({len(gvp_only)} params)")
 if mol_only: 
 torch.save(mol_only, str(extras_save_dir / "mol_adapter.pt"))
 print(f"✅ save Mol adapter ({len(mol_only)} params)")
 if diff_only: 
 torch.save(diff_only, str(extras_save_dir / "diffusion_adapter.pt"))
 print(f"✅ save Diffusion adapter ({len(diff_only)} params)")

 print(f"\n🎉 completeoutputdirectory: {output_dir}")
 return True


if __name__ == "__main__":
 # defaultconfigforrun
 # epoch2_1 trainingcompletelast checkpoint
 checkpoint_path = "${CHECKPOINT_DIR:-/path/to/checkpoints}/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/name_conversion/checkpoint-535"
 output_dir = ""
 if len(sys.argv) > 1:
 checkpoint_path = sys.argv[1]
 if len(sys.argv) > 2:
 output_dir = sys.argv[2]
 else:
 output_dir = checkpoint_path
 split_checkpoint(checkpoint_path, output_dir)

