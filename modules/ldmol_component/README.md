## LDMol Component

 LDMol `ldmol_config.yaml`

### 

| | |
|------|------|
| **LDMolInferer** | T2MDDS |
| **LDMolTrainer** | DDP |

---

## 

```
ldmol_component/
├── LDMolInferer.py # 
├── LDMolTrainer.py # 
├── ldmol_config.yaml # 
├── __init__.py # 
├── __main__.py # 
├── test_inferer.sh # 
├── test_trainer.sh # 
├── utils.py # 
├── assets/ # 
│ ├── config_decoder.json
│ ├── config_encoder.json
│ └── vocab_bpe_300_sc.txt
├── diffusion/ # Diffusion 
├── DiT/ # DiT 
└── autoencoder/ # Autoencoder 
```

---

## (LDMolInferer)

### 

`ldmol_config.yaml` 

| | |
|--------|------|
| `text_encoder_name` | `qwen` |
| `text_encoder_path` | Text Encoder |
| `ldmol_ckpt_path` | DiT checkpoint |
| `vae_ckpt_path` | Autoencoder checkpoint |
| `num_sampling_steps` | T2M 100 |
| `cfg_scale` | T2M CFG 2.5 |
| `dds_*` | DDS |

### API 

```python
from ldmol_component import LDMolInferer

# text_encoder
ldmol = LDMolInferer(device="cuda:0")

# T2M
smiles = ldmol.generate_smi_t2m(
 description="a drug-like small molecule with high solubility..."
)

# DDS
new_smiles = ldmol.generate_smi_dds(
 input_smiles="CN1CCc2nc(O)n3nc(-c4ccccc4Cl)nc3c2C1",
 source_text="This molecule has low permeability.",
 target_text="This molecule has improved permeability."
)
```

### LLM + Diffusion

 Qwen hidden states SMILES LLM + Diffusion 

```python
from ldmol_component import LDMolInferer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# LDMol
ldmol = LDMolInferer(device="cuda:0")

# Qwen
qwen = AutoModelForCausalLM.from_pretrained("/path/to/qwen", torch_dtype=torch.bfloat16).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained("/path/to/qwen")

# 1
smiles = ldmol.generate_molecule(
 description="a drug-like molecule with improved solubility...",
 qwen=qwen, # Qwen
 qwen_tokenizer=tokenizer, # tokenizer
)

# 2 hidden states 
# Qwen hidden states
y_cond = torch.randn(1, 512, 4096, device="cuda:0") # (B, L, hidden_dim)
pad_mask = torch.ones(1, 512, device="cuda:0") # (B, L)

smiles_list = ldmol.generate_smi_from_hidden(
 y_cond=y_cond,
 pad_mask=pad_mask,
)
```

`scripts/drug_optim/code/llm_diffusion_cotrain/README.md`

### 

```bash
cd LDMol
bash ldmol_component/test_inferer.sh

# 
DEVICE=cuda:1 bash ldmol_component/test_inferer.sh
```

---

## (LDMolTrainer)

### 

 TSV 

```
SMILES\t
```



```
CID\tSMILES\t
```

### API 

```python
from ldmol_component import LDMolTrainer, TrainConfig

# 
config = TrainConfig(
 data_path="./data/train.txt",
 text_encoder_path="/path/to/qwen3_8b",
 vae_ckpt_path="/path/to/vae.ckpt",
 epochs=100,
 global_batch_size=64,
)

# 
trainer = LDMolTrainer(config)
trainer.train()
```

### 

```bash
# 
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
 -m ldmol_component.LDMolTrainer \
 --data_path ./data/train.txt \
 --text_encoder_path /path/to/qwen3_8b \
 --vae_ckpt_path /path/to/vae.ckpt \
 --epochs 100 \
 --global_batch_size 64

# 4 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
 -m ldmol_component.LDMolTrainer \
 --data_path ./data/train.txt \
 --text_encoder_path /path/to/qwen3_8b \
 --vae_ckpt_path /path/to/vae.ckpt \
 --epochs 100 \
 --global_batch_size 128
```

### 

```bash
cd LDMol

# 
bash ldmol_component/test_trainer.sh

# 
GPUS=0,1,2,3 NPROC=4 bash ldmol_component/test_trainer.sh

# 
DATA_PATH=./data/my_train.txt \
EPOCHS=50 \
GLOBAL_BATCH_SIZE=128 \
bash ldmol_component/test_trainer.sh
```

### 

| | | |
|------|--------|------|
| `data_path` | - | |
| `vae_ckpt_path` | - | VAE |
| `text_encoder_path` | - | Text Encoder |
| `ldmol_ckpt_path` | "" | DiT |
| `epochs` | 100 | |
| `global_batch_size` | 64 | batch size GPU |
| `learning_rate` | 1e-4 | |
| `description_length` | 256 | |
| `results_dir` | "./results" | |
| `log_every` | 100 | |
| `ckpt_every` | 5000 | |

### 

```
results/
└── 000-LDMol/
 ├── log.txt # 
 └── checkpoints/
 ├── 0005000.pt # step 5000 checkpoint
 ├── 0010000.pt # step 10000 checkpoint
 └── ...
```

---

## 

1. **Text Encoder** `qwen` `AssertionError`
2. ****
3. **DDP** `torchrun` 
4. ****Qwen3_8B 80GB GPU
