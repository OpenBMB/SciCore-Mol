<div align="center">
<h1>SciCore-Mol: Augmenting Large Language Models with Pluggable Molecular Cognition Modules</h1>

<a href='#'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='#'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>

**Yuxuan Chen**<sup>1</sup>,
**Changwei Lv**<sup>1</sup>,
**Yunduo Xiao**<sup>1</sup>,
**Wei Wang**<sup>1</sup>,
**Li Jin**<sup>1</sup>,
**Yukun Yan**<sup>1</sup>,
**Zheni Zeng**<sup>1†</sup>,
**Zhiyuan Liu**<sup>1</sup>

<sup>1</sup>Tsinghua University &nbsp;&nbsp; <sup>†</sup>Corresponding Author

</div>

## 📖 Introduction

Large language models (LLMs) are increasingly popular in professional domains, while meet a fundamental cognitive tension when dealing with heterogeneous scientific data: LLMs are designed for discrete natural language symbolic sequences, whereas scientific entities represented by molecules are inherently topological and geometric. Forcing these structures into linear text inevitably results in information loss and semantic noise interferes with the LLM's cognitive reasoning.

We propose **SciCore-Mol**, a novel paradigm to augment the LLM with pluggable external cognitive modules, including a **GVP encoder**, a **diffusion generator**, and a **numerical-sensitive Transformer**. This architecture preserves the general capabilities while provides specialized molecular perception for LLMs. With a two-stage alignment mechanism, external modules are invoked via special tokens and fused at the hidden-state level, enabling the LLM to deeply understand molecular information without sacrificing its core reasoning process.

<p align="center"><img src="figs/fig2.pdf" width="85%"></p>

## ⚙️ Setup

### Prerequisites

- Python 3.10
- CUDA 12.1
- 8x A800 80GB GPUs (recommended for training)

### Installation

```bash
git clone https://github.com/ChenYX24/SciCore-Mol.git
cd SciCore-Mol

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Optional: install flash attention
pip install -e ".[flashattn]"

# Optional: install graph neural network dependencies
pip install -e ".[graph]"

# Optional: install training dependencies
pip install -e ".[train]"
```

### Environment Variables

Copy and configure the environment template:

```bash
cp configs/env.example.sh configs/env.sh
source configs/env.sh
```

Key variables to set:
- `SCICORE_ROOT`: Project root directory
- `MODEL_DIR`: Path to base models (e.g., Qwen3-8B)
- `CHECKPOINT_DIR`: Path to trained checkpoints
- `DATA_DIR`: Path to training/evaluation data
- `OPENAI_API_KEY`: API key for GPT baseline evaluation

## 🔧 Training

### Stage 1: LLM SFT with Molecular Awareness

```bash
# Configure paths in configs/cotrain_layer2_llm_v3.yaml
bash scripts/train/train.sh
```

### Stage 2: Diffusion Data Generation

```bash
bash cotrain_llm_diffusion/run_generate_v3.sh
```

### Stage 3: Layer2 Training (Numerical-Sensitive Transformer)

```bash
python scripts/layer2/train_layer2.py --config scripts/layer2/layer2_train_config_stage2_v7b.yaml
```

## 📊 Evaluation

### ChemBench4K

```bash
bash scripts/run/run_chembench_all_tasks.sh
```

### MMLU Chemistry Subsets

```bash
python scripts/eval/eval_mmlu_interns1mini_5subsets.py \
    --model_path ${MODEL_DIR}/your-model \
    --output_dir eval_results/mmlu/
```

### ORD Reaction Prediction

```bash
bash scripts/layer2_llm/run_full_pipeline.sh
```

### SMolInstruct

```bash
bash scripts/run/eval_smol_task_list.sh
```

## 📁 Repository Structure

```
SciCore-Mol/
├── configs/                    # Training and evaluation configs
│   ├── cotrain_layer2_llm_*.yaml
│   ├── deepspeed_*.json
│   └── env.example.sh
├── cotrain_llm_diffusion/      # Stage 1 & 2: LLM-Diffusion co-training
│   ├── train_step1_llm.py
│   └── generate_reasoning*.py
├── eval/                       # Evaluation scripts
│   ├── drug_optim/             # Drug optimization evaluation
│   └── eval_*.py               # Benchmark evaluations
├── modules/                    # Core model components
│   ├── mol_aware_lm.py         # Molecular-aware language model
│   ├── model_init.py           # Model initialization
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── layer2_component/       # Layer2: numerical-sensitive Transformer
│   ├── ldmol_component/        # LDMol: diffusion-based molecule generator
│   └── tools.py                # Chemical entity extraction & SMILES tools
├── scripts/
│   ├── train/                  # Training scripts
│   ├── eval/                   # Evaluation scripts
│   ├── layer2/                 # Layer2 training configs and scripts
│   ├── layer2_llm/             # Layer2-LLM integration pipeline
│   ├── postprocess/            # Result post-processing and scoring
│   ├── preprocess/             # Data preprocessing
│   └── ckpt/                   # Checkpoint management utilities
├── utils/                      # Shared utilities
│   ├── metrics.py              # Evaluation metrics
│   └── smiles_canonicalization.py
├── vendor/                     # Third-party dependencies
│   └── gvp-pytorch-main/       # GVP-GNN implementation
├── figs/                       # Paper figures
├── pyproject.toml              # Project configuration
└── README.md
```

## 📄 Acknowledgement

- [GVP-GNN](https://github.com/drorlab/gvp-pytorch) — Geometric Vector Perceptron for molecular structure encoding
- [LDMol](https://github.com/jinhojsk515/LDMol) — Latent Diffusion for Molecular generation
- [SMolInstruct](https://github.com/osu-nlp-group/SMolInstruct) — Molecular instruction tuning benchmark
- [ChemBench](https://github.com/lamalab-org/chem-bench) — Chemistry benchmark suite

## 🥰 Citation

```bibtex
@article{chen2025scicoremol,
  title={SciCore-Mol: Augmenting Large Language Models with Pluggable Molecular Cognition Modules},
  author={Chen, Yuxuan and Lv, Changwei and Xiao, Yunduo and Wang, Wei and Jin, Li and Yan, Yukun and Zeng, Zheni and Liu, Zhiyuan},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## 📧 Contact

If you have questions, suggestions, or bug reports, please open an issue or email:
```
yxchen0524@gmail.com
```

## 📜 License

This project is dual-licensed under [MIT](LICENSE-MIT) and [Apache 2.0](LICENSE-APACHE). You may choose either license at your option.
