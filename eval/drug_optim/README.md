# Drug Optimization 

LLM Diffusion

## 

```bash
cd ${SCICORE_ROOT:-/path/to/scicore-mol}/eval/drug_optim

# LLM 
python run_drug_optim_eval.py --config config/llm_cpt_sft.yaml

# Diffusion 
python run_drug_optim_eval.py --config config/diffusion_base.yaml
```

## 

```
eval/drug_optim/
├── run_drug_optim_eval.py # 
├── config/ # 
│ ├── llm_base.yaml
│ ├── llm_cpt_sft.yaml
│ └── diffusion_base.yaml
├── testers/ # 
│ ├── base.py
│ ├── llm_tester.py
│ └── diffusion_tester.py
├── scoring/ # 
│ ├── scorer.py # 
│ ├── admet_reasoning_richness.py
│ ├── filter.py
│ └── test2.py
└── eval_output/ # 
 └── <model_name>/
 ├── output.txt
 ├── test_log.log
 ├── scoring_summary.json
 └── run_info.json
```

## 

### LLM 

```yaml
model_type: llm
model_name: llm_cpt_sft
ckpt: /path/to/checkpoint
input_data: /path/to/test_text2smi.jsonl
algorithm: chat
device: auto # auto / cuda:0 / 0 / 0,1,2
max_new_tokens: 256
temperature: 0.7
```

### Diffusion 

```yaml
model_type: diffusion
model_name: diffusion_base
ckpt: /path/to/checkpoint.pt
input_data: /path/to/test_dds.txt
algorithm: dds
device: cuda:0 # cuda:0 / 0
```

### 

| | |
|----|------|
| `auto` | GPU |
| `cuda:0` | |
| `0` | `cuda:0` |
| `0,1,2` | LLM CUDA_VISIBLE_DEVICES |

## 

```
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ │────▶│ │────▶│ │
│ config/*.yaml │ │ output.txt │ │ ADMET │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

****
- LLM: chat SMILES `test_text2smi.jsonl` prompt
- Diffusion: DDS `test_dds.txt` 

****
- `scoring/admet_reasoning_richness.py` ADMET 
- `scoring_summary.json`

## 

### output.txt

Tab 6 

| | |
|------|------|
| row_id | |
| original_smiles | |
| source_caption | |
| target_caption | |
| gt_smiles | Ground truth SMILES |
| pred_smiles | SMILES |

### scoring_summary.json

 ADMET 
- `avg_main_reward`: 
- `avg_bonus_f1`: F1 
- `validity_rate`: 
- ...

## 

| | |
|------|------|
| `--config` | |
| `--output-dir` | |
| `--skip-test` | |
| `--skip-score` | |

## 

| | |
|------|------|
| llm_base | `${DATA_DIR:-/path/to/data}/base_model/qwen3_8b` |
| llm_cpt_sft | `${CHECKPOINT_DIR:-/path/to/checkpoints}/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/checkpoint-4200` |
| diffusion_base | `${CHECKPOINT_DIR:-/path/to/checkpoints}/diffusion_pretrained/ours/ldmol/ldmol_chatmol.pt` |

## 

| | | |
|----------|----------|------|
| LLM | `test_text2smi.jsonl` | JSONL prompt/ground_truth |
| Diffusion | `test_dds.txt` | TSV4 |

`${SCICORE_ROOT:-/path/to/scicore-mol}/eval_results/data/ldmol/drug_optim/processed/`
