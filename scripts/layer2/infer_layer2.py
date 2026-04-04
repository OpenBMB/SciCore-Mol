#!/usr/bin/env python3
"""
inference Layer2 model
: python scripts/layer2/infer_layer2.py --input data.jsonl --output predictions.jsonl
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# directorypath
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.layer2_component import Layer2Inferer


def main():
 # defaultpath
 DEFAULT_CONFIG = "${SCICORE_ROOT:-/path/to/scicore-mol}/modules/layer2_component/layer2_config.yaml"
 DEFAULT_GVP_CKPT = "${CHECKPOINT_DIR:-/path/to/checkpoints}/gvp_weights_best.pt"
 DEFAULT_INPUT = "${SCICORE_ROOT:-/path/to/scicore-mol}/Layer2/data/test.jsonl"
 DEFAULT_OUTPUT = "${SCICORE_ROOT:-/path/to/scicore-mol}/scripts/layer2/data/predictions.jsonl"
 
 parser = argparse.ArgumentParser(description="Layer2 inference")
 parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help=f"inputfileJSONLcontains reactant_smilesdefault: {DEFAULT_INPUT}")
 parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help=f"outputfileJSONLdefault: {DEFAULT_OUTPUT}")
 parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help=f"Layer2 configfilepathdefault: {DEFAULT_CONFIG}")
 parser.add_argument("--device", type=str, default="cuda:0", help="devicedefault: cuda:0")
 parser.add_argument("--gvp_ckpt", type=str, default=DEFAULT_GVP_CKPT, help=f"GVP checkpoint pathdefault: {DEFAULT_GVP_CKPT}")
 
 args = parser.parse_args()
 
 # initialize Layer2Inferer
 print("📦 initialize Layer2Inferer...")
 inferer = Layer2Inferer(
 config_path=args.config,
 device=args.device,
 gvp_ckpt_path=args.gvp_ckpt,
 )
 
 # loadinput data
 print(f"📂 loadinput: {args.input}")
 inputs = []
 with open(args.input, 'r', encoding='utf-8') as f:
 for line in f:
 line = line.strip()
 if not line:
 continue
 try:
 data = json.loads(line)
 inputs.append(data)
 except json.JSONDecodeError:
 inputs.append({"reactant_smiles": line})
 
 print(f" {len(inputs)} data")
 
 # inference
 print("🔄 startinference...")
 results = []
 for i, item in enumerate(inputs):
 reactant_smiles = item.get("reactant_smiles", "")
 if not reactant_smiles:
 continue
 
 try:
 # prediction
 output = inferer.predict(reactant_smiles=reactant_smiles)
 
 result = {
 "reactant_smiles": reactant_smiles,
 "yield_bin": int(output['yield_bin']),
 "yield_reg": float(output['yield_reg']),
 "embedding": output['embedding'].cpu().tolist(),
 }
 results.append(result)
 
 if (i + 1) % 100 == 0:
 print(f" process {i + 1}/{len(inputs)} ")
 
 except Exception as e:
 print(f"❌ process {i} data: {e}")
 continue
 
 # saveresult
 print(f"💾 saveresult: {args.output}")
 with open(args.output, 'w', encoding='utf-8') as f:
 for result in results:
 f.write(json.dumps(result, ensure_ascii=False) + '\n')
 
 print(f"✅ completeprocess {len(results)} data")


if __name__ == "__main__":
 main()
