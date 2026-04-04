#!/usr/bin/env python3
"""
Multi-granularity yield classification evaluation.

Evaluates yield prediction at 3-class, 5-class, and 10-class granularity
by merging the 10-bin logits into coarser bins.

  10-class: bins 0-9 (each 10%)
   5-class: [0,1] [2,3] [4,5] [6,7] [8,9] (each 20%)
   3-class: [0,1,2] [3,4,5,6] [7,8,9] (Low/Mid/High)

Usage:
    cd ${SCICORE_ROOT:-/path/to/scicore-mol}
    python scripts/layer2/eval_yield_multiclass.py \
        --checkpoints v4=/path/to/v4.pt v5=/path/to/v5.pt \
        --data ${SCICORE_ROOT:-/path/to/scicore-mol}/Layer2/data/ord_layer2_v2/layer2_test.jsonl
"""

import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from modules.layer2_component.model import ModelConfig, Layer2PretrainModel
from modules.layer2_component.dataset import Layer2JsonlIndexed
from modules.layer2_component.collate import collate_layer2
from modules.layer2_component.masking import EvalMaskingConfig


# --- Bin merging schemes ---
# 5-class: pairs of 2
MERGE_5 = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
LABELS_5 = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]

# 3-class: Low / Mid / High
MERGE_3 = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]
LABELS_3 = ["Low(0-30%)", "Mid(30-70%)", "High(70-100%)"]


def load_model(checkpoint_path: str, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    config = ckpt.get("config", ckpt.get("cfg", {}))
    if isinstance(config, dict):
        model_cfg = ModelConfig(
            mol_emb_dim=config.get("mol_emb_dim", 256),
            hidden_dim=config.get("hidden_dim", 768),
            n_layers=config.get("n_layers", 8),
            n_heads=config.get("n_heads", 12),
            dropout=config.get("dropout", 0.2),
            num_roles=config.get("num_roles", 11),
            num_token_types=config.get("num_token_types", 2),
            tau=config.get("tau", 0.07),
            learnable_tau=config.get("learnable_tau", False),
            symmetric_ince=config.get("symmetric_ince", False),
            use_projection_head=config.get("use_projection_head", False),
            head_dropout=config.get("head_dropout", 0.0),
        )
    else:
        model_cfg = config

    model = Layer2PretrainModel(model_cfg)

    if "model" in ckpt:
        sd = ckpt["model"]
    elif "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt

    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


@torch.no_grad()
def extract_yield_logits(model, data_path: str, batch_size: int = 64, num_workers: int = 2):
    ds = Layer2JsonlIndexed(data_path, masking=True, masking_cfg=EvalMaskingConfig())
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_layer2,
                        num_workers=num_workers, shuffle=False)

    all_logits = []
    all_labels = []
    all_reg_pred = []
    all_reg_true = []

    for batch in loader:
        out = model(batch)
        y_mask = batch.yield_pred_mask
        idx = (y_mask > 0.5).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        logits = out["pred_yield_bin"][idx].float().cpu()
        labels = batch.yield_bin[idx].cpu()
        reg_pred = out["pred_yield_reg"][idx].float().cpu()
        reg_true = batch.yield_reg[idx].cpu()

        all_logits.append(logits)
        all_labels.append(labels)
        all_reg_pred.append(reg_pred)
        all_reg_true.append(reg_true)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    reg_pred = torch.cat(all_reg_pred, dim=0)
    reg_true = torch.cat(all_reg_true, dim=0)
    return logits, labels, reg_pred, reg_true


def merge_and_eval(logits_10, labels_10, merge_scheme, class_labels):
    """Merge 10-bin probabilities into coarser classes and compute accuracy."""
    probs = torch.softmax(logits_10, dim=-1)  # [N, 10]
    n_classes = len(merge_scheme)

    merged_probs = torch.zeros(probs.shape[0], n_classes)
    for c, bins in enumerate(merge_scheme):
        for b in bins:
            merged_probs[:, c] += probs[:, b]

    merged_preds = merged_probs.argmax(dim=-1)

    label_map = {}
    for c, bins in enumerate(merge_scheme):
        for b in bins:
            label_map[b] = c
    merged_labels = torch.tensor([label_map[l.item()] for l in labels_10])

    acc = (merged_preds == merged_labels).float().mean().item()
    relaxed = ((merged_preds - merged_labels).abs() <= 1).float().mean().item()

    per_class = {}
    for c in range(n_classes):
        mask = merged_labels == c
        n = mask.sum().item()
        if n > 0:
            per_class[class_labels[c]] = {
                "acc": (merged_preds[mask] == c).float().mean().item(),
                "count": n,
            }
        else:
            per_class[class_labels[c]] = {"acc": float("nan"), "count": 0}

    return acc, relaxed, per_class


def eval_10class(logits, labels):
    preds = logits.argmax(dim=-1)
    acc = (preds == labels).float().mean().item()
    relaxed = ((preds - labels).abs() <= 1).float().mean().item()

    per_class = {}
    for c in range(10):
        mask = labels == c
        n = mask.sum().item()
        lbl = f"{c*10}-{(c+1)*10}%"
        if n > 0:
            per_class[lbl] = {
                "acc": (preds[mask] == c).float().mean().item(),
                "count": n,
            }
        else:
            per_class[lbl] = {"acc": float("nan"), "count": 0}
    return acc, relaxed, per_class


def print_results(name, acc, relaxed, per_class, n_classes):
    print(f"\n  [{n_classes}-class] Accuracy: {acc:.4f}  Relaxed(+/-1): {relaxed:.4f}")
    for lbl, info in per_class.items():
        a = f"{info['acc']:.3f}" if not np.isnan(info["acc"]) else "N/A"
        print(f"    {lbl:15s}  acc={a}  n={info['count']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="name=path pairs, e.g. v4=/path/to/ckpt.pt")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    ckpts = {}
    for spec in args.checkpoints:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            name = Path(spec).stem
            path = spec
        ckpts[name] = path

    print(f"Test data: {args.data}")
    print(f"Models to evaluate: {list(ckpts.keys())}")

    for name, path in ckpts.items():
        print(f"\n{'='*60}")
        print(f"Model: {name}")
        print(f"Checkpoint: {path}")
        print("=" * 60)

        model = load_model(path)
        logits, labels, reg_pred, reg_true = extract_yield_logits(
            model, args.data, args.batch_size, args.num_workers)
        print(f"Yield samples: {logits.shape[0]}")

        # Regression metrics
        mae = (reg_pred - reg_true).abs().mean().item()
        rmse = ((reg_pred - reg_true) ** 2).mean().sqrt().item()
        print(f"Regression:  MAE={mae:.4f}  RMSE={rmse:.4f}")

        # 10-class
        acc10, rel10, pc10 = eval_10class(logits, labels)
        print_results(name, acc10, rel10, pc10, 10)

        # 5-class
        acc5, rel5, pc5 = merge_and_eval(logits, labels, MERGE_5, LABELS_5)
        print_results(name, acc5, rel5, pc5, 5)

        # 3-class
        acc3, rel3, pc3 = merge_and_eval(logits, labels, MERGE_3, LABELS_3)
        print_results(name, acc3, rel3, pc3, 3)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
