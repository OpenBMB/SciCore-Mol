# Multimodal Molecule Editor

 LDMol GVP 

## 

```
: source_smiles + edit_instruction
: target_smiles ()
```

## 

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Multimodal Molecule Editor │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ │
│ ┌──────────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ INPUT PROCESSING │ │
│ └──────────────────────────────────────────────────────────────────────────────────────────┘ │
│ │
│ source_smiles edit_instruction │
│ │ │ │
│ ▼ ▼ │
│ ┌───────────────┐ ┌────────────────────┐ │
│ │ RDKit 3D │ │ Tokenizer │ │
│ │ Embedding │ │ (Qwen3) │ │
│ └───────┬───────┘ └──────────┬─────────┘ │
│ │ │ │
│ ▼ ▼ │
│ ┌───────────────────────┐ ┌────────────────────────────┐ │
│ │ GVP │ │ Text Encoder │ │
│ │ ┌─────────────────┐ │ │ ┌──────────────────────┐ │ │
│ │ │ Node: atoms │ │ │ │ Qwen3-8B (frozen) │ │ │
│ │ │ Edge: bonds │ │ │ │ │ │ │
│ │ │ Coord: 3D pos │ │ │ │ hidden_size: 4096 │ │ │
│ │ └─────────────────┘ │ │ └──────────────────────┘ │ │
│ │ │ │ │ │
│ │ output_dim: 256 │ │ output: [B, L, 4096] │ │
│ │ (frozen) │ │ (frozen) │ │
│ └───────────┬───────────┘ └─────────────┬──────────────┘ │
│ │ │ │
│ │ [B, 256] │ [B, L, 4096] │
│ │ │ │
│ ┌───────────┴─────────────────────────────────────────────────┴───────────┐ │
│ │ │ │
│ │ CONDITION FUSION MODULE (trainable) │ │
│ │ ┌────────────────────────────────────────────────────────────────┐ │ │
│ │ │ │ │ │
│ │ │ GVP [B, 256] ──→ Linear(256, 1152) ──→ [B, 1, 1152] │ │ │
│ │ │ │ │ │ │
│ │ │ │ concat │ │ │
│ │ │ ▼ │ │ │
│ │ │ Text [B, L, 4096] ──→ Linear(4096, 1152) ──→ [B, L, 1152] │ │ │
│ │ │ │ │ │ │
│ │ │ ▼ │ │ │
│ │ │ [B, L+1, 1152] ───────────────────→ y_cond │
│ │ │ │ │ │
│ │ └────────────────────────────────────────────────────────────────┘ │ │
│ │ │ │
│ └──────────────────────────────────────────────────────────────────────────┘ │
│ │
│ ┌──────────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ DIFFUSION PROCESS │ │
│ └──────────────────────────────────────────────────────────────────────────────────────────┘ │
│ │
│ y_cond [B, L+1, 1152] │
│ │ │
│ ▼ │
│ x_T ──────────→ ┌─────────────────────────────────┐ │
│ [B, 64, 127, 1] │ DiT │ │
│ (random noise) │ ┌───────────────────────────┐ │ │
│ │ │ Transformer Blocks (28) │ │ │
│ │ │ - Self-Attention │ │ │
│ │ │ - Cross-Attention ← cond │ │ │
│ │ │ - MLP │ │ │
│ │ │ │ │ │
│ │ │ LoRA fine-tune (rank=8) │ │ │
│ │ └───────────────────────────┘ │ │
│ │ │ │
│ │ DDPM Sampling (50 steps) │ │
│ └────────────────┬────────────────┘ │
│ │ │
│ ▼ │
│ x_0 [B, 64, 127, 1] │
│ (denoised latent) │
│ │
│ ┌──────────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ DECODING │ │
│ └──────────────────────────────────────────────────────────────────────────────────────────┘ │
│ │
│ x_0 [B, 64, 127, 1] │
│ │ │
│ ▼ │
│ ┌─────────────────────────────────┐ │
│ │ AE Decoder (frozen) │ │
│ │ ┌───────────────────────────┐ │ │
│ │ │ reshape → [B, 127, 64] │ │ │
│ │ │ decode_prefix(64→1024) │ │ │
│ │ │ BERT Decoder │ │ │
│ │ │ vocab_size: 300 │ │ │
│ │ └───────────────────────────┘ │ │
│ └────────────────┬────────────────┘ │
│ │ │
│ ▼ │
│ target_smiles │
│ │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ TRAINING DATAFLOW │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ │
│ train_mol_edit.jsonl │
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │ { │ │
│ │ "source_smiles": "CCCc1c(OC)nc2nc(C(=O)c3ccc(Cl)cc3)cn2c1CC", │ │
│ │ "edit_instruction": "Optimize ... to reduce high DILI risk ...", │ │
│ │ "target_smiles": "CCCc1c(OC)nc2nc(C(=O)c3ccccc3)cn2c1CC", │ │
│ │ "template": "natural_language" │ │
│ │ } │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│ │ │ │ │
│ ▼ ▼ ▼ │
│ ┌───────────┐ ┌────────────┐ ┌─────────────┐ │
│ │ GVP │ │ Qwen3 │ │ AE Encoder │ │
│ └─────┬─────┘ └──────┬─────┘ └──────┬──────┘ │
│ │ │ │ │
│ ▼ ▼ ▼ │
│ gvp_emb [B,256] text_emb [B,L,4096] x_0 [B,64,127,1] │
│ │ │ │ │
│ └────────────┬───────────┘ │ │
│ ▼ │ │
│ Fusion │ │
│ │ │ │
│ ▼ │ │
│ y_cond [B,L+1,1152] │ │
│ │ │ │
│ └──────────────────┬─────────────────┘ │
│ │ │
│ ▼ │
│ ┌─────────┐ │
│ │ DiT │ │
│ │ Denoise │ │
│ └────┬────┘ │
│ │ │
│ ▼ │
│ L_diffusion │
│ │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 

| | | | | |
|------|------|------|------|------|
| **GVP** | | frozen | source_smiles → 3D graph | `[B, 256]` |
| **Text Encoder** | Qwen3-8B | frozen | edit_instruction | `[B, L, 4096]` |
| **Condition Fusion** | | trainable | GVP emb + Text emb | `[B, L', hidden]` |
| **DiT** | LDMol | fine-tune (LoRA) | x_t + y_cond | x_0 |
| **AE Decoder** | LDMol | frozen | latent | SMILES |

## Condition Fusion Module

 GVP Text 

```
GVP: [B, 256] ──── proj ────→ [B, 1, hidden] ─┐
 ├──→ concat ──→ [B, L+1, hidden]
Text: [B, L, 4096] ─ proj ───→ [B, L, hidden] ─┘
```

## 

****GVP, Text Encoder, AE Encoder/Decoder 
****DiT (LoRA), Condition Fusion Module

****
```
L = L_diffusion(x_0, x̂_0) + λ · L_recon(target_smiles, decoded_smiles)
```

## 

| | | |
|------|------|------|
| `full_context` | ~77% | instruction + ADMET profile |
| `natural_language` | ~14% | |
| `reasoning_guided` | ~9% | |

`eval_results/data/ldmol/drug_optim/processed/train_mol_edit.jsonl`

## 

```python
# 1. 
gvp_emb = gvp_encoder(source_smiles) # [B, 256]
text_emb = text_encoder(edit_instruction) # [B, L, 4096]
y_cond = fusion_module(gvp_emb, text_emb) # [B, L', hidden]

# 2. 
x_T = torch.randn(B, C, 127, 1)
x_0 = dit.sample(x_T, y_cond, steps=50)

# 3. 
target_smiles = ae_decoder(x_0)
```

## 

```
modules/ldmol_component/
├── multimodal_molecule_editor.md # 
├── molecule_editor.py # MoleculeEditor 
├── condition_fusion.py # ConditionFusionModule
└── LDMolTrainer.py # 
```
