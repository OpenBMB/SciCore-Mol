#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model .bin formatconvert safetensors format
ifmodeldirectory .bin fileconvert safetensors
"""

import argparse
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def check_has_safetensors(model_path: str) -> bool:
 """checkmodeldirectorywhether safetensors file"""
 model_dir = Path(model_path)
 if not model_dir.exists():
 return False
 
 safetensors_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("model*.safetensors"))
 return len(safetensors_files) > 0


def check_has_bin_files(model_path: str) -> bool:
 """checkmodeldirectorywhether .bin file"""
 model_dir = Path(model_path)
 if not model_dir.exists():
 return False
 
 bin_files = list(model_dir.glob("*.bin")) + list(model_dir.glob("pytorch_model*.bin"))
 return len(bin_files) > 0


def convert_to_safetensors(model_path: str, device: str = "cpu", dtype: str = "bfloat16"):
 """
 modelconvert safetensors format
 
 Args:
 model_path: model path
 device: loaddevicedefault cpuGPU memory
 dtype: datatypebfloat16 float16
 """
 print(f"📂 Checking model: {model_path}")
 
 # checkwhether safetensors
 if check_has_safetensors(model_path):
 print("✅ Model already has safetensors files. Skipping conversion.")
 return
 
 # checkwhether .bin file
 if not check_has_bin_files(model_path):
 print("⚠️ No .bin files found. Nothing to convert.")
 return
 
 print("🔄 Converting .bin files to safetensors format...")
 print(f" Device: {device}")
 print(f" Dtype: {dtype}")
 
 # choose dtype
 if dtype == "bfloat16":
 torch_dtype = torch.bfloat16
 elif dtype == "float16":
 torch_dtype = torch.float16
 else:
 torch_dtype = torch.float32
 
 try:
 # loadmodeluse CPU deviceGPU memory
 print(" Loading model...")
 model = AutoModelForCausalLM.from_pretrained(
 model_path,
 torch_dtype=torch_dtype,
 low_cpu_mem_usage=True,
 device_map=device if device != "cpu" else None
 )
 
 # if GPUneedsdevice
 if device.startswith("cuda"):
 model = model.to(device)
 
 print(" Saving as safetensors...")
 # save safetensors format
 model.save_pretrained(
 model_path,
 safe_serialization=True,
 max_shard_size="5GB" # ifmodelsave
 )
 
 print("✅ Conversion completed successfully!")
 print(f" Safetensors files saved to: {model_path}")
 
 # optionaldeleteold .bin file
 # print(" Cleaning up old .bin files...")
 # for bin_file in Path(model_path).glob("*.bin"):
 # if "pytorch_model" in bin_file.name:
 # bin_file.unlink()
 # print(f" Deleted: {bin_file.name}")
 
 except Exception as e:
 print(f"❌ Conversion failed: {e}")
 import traceback
 traceback.print_exc()
 raise


def main():
 parser = argparse.ArgumentParser(description="Convert model from .bin to safetensors format")
 parser.add_argument('--model-path', type=str, required=True,
 help='Path to the model directory')
 parser.add_argument('--device', type=str, default="cpu",
 help='Device to load model (cpu, cuda:0, etc.). Default: cpu to save memory')
 parser.add_argument('--dtype', type=str, default="bfloat16",
 choices=["bfloat16", "float16", "float32"],
 help='Data type for model weights. Default: bfloat16')
 parser.add_argument('--check-only', action='store_true',
 help='Only check if conversion is needed, do not convert')
 
 args = parser.parse_args()
 
 if args.check_only:
 has_safetensors = check_has_safetensors(args.model_path)
 has_bin = check_has_bin_files(args.model_path)
 
 print(f"Model: {args.model_path}")
 print(f" Has safetensors: {has_safetensors}")
 print(f" Has .bin files: {has_bin}")
 
 if has_safetensors:
 print("✅ No conversion needed.")
 elif has_bin:
 print("⚠️ Conversion needed: model has .bin files but no safetensors.")
 else:
 print("⚠️ No model files found.")
 else:
 convert_to_safetensors(args.model_path, args.device, args.dtype)


if __name__ == "__main__":
 main()

