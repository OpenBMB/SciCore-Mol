#!/usr/bin/env python3
"""
debugscripttest sft_tester.py differentconfig
supports
1. generateuse GVP diffusion
2. +gvpuse GVP
3. +gvp+diffusionuse GVP diffusion supplement
4. task diffusiongeneration
"""

import sys
import os
import argparse
from pathlib import Path

# directorypath
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sft_tester import MolAwareGenerator2

# defaultconfig
DEFAULT_CONFIG = {
 "ckpt_dir": "${CHECKPOINT_DIR:-/path/to/checkpoints}/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/checkpoint-4200",
 "device": "cuda:0",
 "device_map": None, # single GPUmode
 "dtype": "bf16",
 "debug": True,
 "token_classifier_path": "${CHECKPOINT_DIR:-/path/to/checkpoints}/qwen3_mlp_token_head.pt",
}

# # Diffusion config
# LDMOL_CONFIG = {
# "enabled": True,
# "ckpt_path": "${CHECKPOINT_DIR:-/path/to/checkpoints}/diffusion_pretrained/ours/ldmol/ldmol_chatmol-qwen3_8b.pt",
# "vae_path": "${CHECKPOINT_DIR:-/path/to/checkpoints}/diffusion_pretrained/official/checkpoint_autoencoder.ckpt",
# "num_sampling_steps": 100,
# "cfg_scale": 2.5,
# }

# test prompt
TEST_PROMPTS = {
 "normal": "Describe this molecule: CCCCCCC(O)C/C=C\\CCCCCCCC(=O)[O-]\nPlease only output the answer.",
 "generation": "Generate a molecule that is a potential drug candidate for treating diabetes. Please only output the answer.",
 "synthesis": "Predict a possible product from the listed reactants and reagents. CCN.CN1C=CC=C1C=O\nPlease only output the answer without any explanation or additional text.",
}


def test_mode_1_direct_generation(cfg, prompt):
 """mode1generateuse GVP diffusion"""
 print("\n" + "="*80)
 print("mode1generateuse GVP diffusion")
 print("="*80)
 
 gen = MolAwareGenerator2()
 gen.load(cfg)
 
 text = gen.generate(
 prompt,
 add_dialog_wrapper=True,
 realtime_mol=False, # use GVP
 max_new_tokens=256,
 do_sample=True,
 temperature=0.2,
 repetition_penalty=1.06, # 
 no_repeat_ngram_size=3, # 3-gram
 skip_special_tokens=True,
 )
 
 print("\n=== Generated Text ===")
 print(text)
 return text


def test_mode_2_with_gvp(cfg, prompt):
 """mode2+gvpuse GVP"""
 print("\n" + "="*80)
 print("mode2+gvpuse GVP")
 print("="*80)
 
 gen = MolAwareGenerator2()
 gen.load(cfg)
 
 text = gen.generate(
 prompt,
 add_dialog_wrapper=True,
 realtime_mol=True, # use GVP
 max_new_tokens=256,
 do_sample=True,
 temperature=0.2,
 repetition_penalty=1.06, # 
 no_repeat_ngram_size=3, # 3-gram
 skip_special_tokens=True,
 )
 
 print("\n=== Generated Text ===")
 print(text)
 return text


def test_mode_3_with_gvp_diffusion(cfg, prompt):
 """mode3+gvp+diffusionuse GVP diffusion supplement"""
 print("\n" + "="*80)
 print("mode3+gvp+diffusionuse GVP diffusion supplement")
 print("="*80)
 
 # NOTE:ldmol config ${SCICORE_ROOT:-/path/to/scicore-mol}/modules/ldmol_component/ldmol_config.yaml
 # cfg_with_ldmol = cfg.copy()
 
 gen = MolAwareGenerator2()
 gen.load(cfg)
 
 text = gen.generate(
 prompt,
 add_dialog_wrapper=True,
 realtime_mol=True, # use GVP
 use_diffusion_as_smiles_supplement=True, # use diffusion supplement
 max_new_tokens=256,
 do_sample=True,
 temperature=0.2,
 repetition_penalty=1.06, # 
 no_repeat_ngram_size=3, # 3-gram
 skip_special_tokens=True,
 )
 
 print("\n=== Generated Text ===")
 print(text)
 return text


def test_mode_4_diffusion_generation(cfg, prompt):
 """mode4task diffusiongeneration"""
 print("\n" + "="*80)
 print("mode4task diffusiongeneration")
 print("="*80)
 
 # NOTE:ldmol config ${SCICORE_ROOT:-/path/to/scicore-mol}/modules/ldmol_component/ldmol_config.yaml
 # cfg_with_ldmol = cfg.copy()
 gen = MolAwareGenerator2()
 gen.load(cfg)
 
 text = gen.generate(
 prompt,
 add_dialog_wrapper=True,
 realtime_mol=True, # use GVP
 task_type="molecule_generation", # taskmoleculegenerateNOTEuse "molecule_generation" diffusion
 max_new_tokens=512,
 do_sample=True,
 temperature=0.2,
 repetition_penalty=1.06, # 
 no_repeat_ngram_size=3, # 3-gram
 skip_special_tokens=True,
 verbose_logging=True, # log diffusion generate
 )
 
 print("\n=== Generated Text ===")
 print(text)
 return text


def main():
 parser = argparse.ArgumentParser(description="debug sft_tester.py differentconfig")
 parser.add_argument(
 "--mode",
 type=int,
 choices=[1, 2, 3, 4],
 default=1,
 help="testmode1=generate, 2=+gvp, 3=+gvp+diffusion, 4=taskdiffusion"
 )
 parser.add_argument(
 "--prompt-type",
 type=str,
 choices=["normal", "generation", "synthesis"],
 default="normal",
 help="Prompt type"
 )
 parser.add_argument(
 "--ckpt-dir",
 type=str,
 default=None,
 help="Checkpoint directorypathdefaultconfig"
 )
 parser.add_argument(
 "--token-classifier",
 type=str,
 default=None,
 help="Token classifier pathdefaultconfig"
 )
 parser.add_argument(
 "--device",
 type=str,
 default="cuda:0",
 help="devicedefaultcuda:0"
 )
 parser.add_argument(
 "--custom-prompt",
 type=str,
 default=None,
 help="define promptdefault prompt"
 )
 
 args = parser.parse_args()
 
 # buildconfig
 cfg = DEFAULT_CONFIG.copy()
 if args.ckpt_dir:
 cfg["ckpt_dir"] = args.ckpt_dir
 if args.token_classifier:
 cfg["token_classifier_path"] = args.token_classifier
 if args.device:
 cfg["device"] = args.device
 
 # choose prompt
 if args.custom_prompt:
 prompt = args.custom_prompt
 else:
 prompt = TEST_PROMPTS[args.prompt_type]
 
 print(f"\nconfig")
 print(f" Checkpoint: {cfg['ckpt_dir']}")
 print(f" Token Classifier: {cfg.get('token_classifier_path', 'None')}")
 print(f" Device: {cfg['device']}")
 print(f" Prompt: {prompt[:100]}...")
 
 # according tomodeexecutetest
 if args.mode == 1:
 test_mode_1_direct_generation(cfg, prompt)
 elif args.mode == 2:
 test_mode_2_with_gvp(cfg, prompt)
 elif args.mode == 3:
 test_mode_3_with_gvp_diffusion(cfg, prompt)
 elif args.mode == 4:
 test_mode_4_diffusion_generation(cfg, prompt)
 
 print("\n" + "="*80)
 print("testcomplete")
 print("="*80)


if __name__ == "__main__":
 main()

