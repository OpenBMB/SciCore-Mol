#!/usr/bin/env python3
"""
Layer2 Model Evaluation Script (fixed)

Standalone eval on test set reporting per-task metrics:
- Embedding: Top-1/5/10 retrieval accuracy, mean cosine similarity
- Yield: bin accuracy (10/5/3 class), softmax-expectation MAE
- Amount: MAE per channel (moles/mass/volume)

Usage:
    cd ${SCICORE_ROOT:-/path/to/scicore-mol}/
    python scripts/layer2/eval_layer2_model.py \
        --checkpoint /path/to/checkpoint.pt \
        --data ${SCICORE_ROOT:-/path/to/scicore-mol}/Layer2/data/ord_layer2_v2/layer2_test.jsonl \
        [--batch_size 64] [--num_workers 4]
"""

import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from modules.layer2_component.model import ModelConfig, Layer2PretrainModel
from modules.layer2_component.dataset import Layer2JsonlIndexed
from modules.layer2_component.collate import collate_layer2
from modules.layer2_component.masking import EvalMaskingConfig

# Bin merging schemes
MERGE_5 = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
MERGE_3 = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]
BIN_CENTERS = torch.arange(10) * 0.1 + 0.05  # [0.05, 0.15, ..., 0.95]


def load_model(checkpoint_path: str, device: str = "cuda:0") -> Layer2PretrainModel:
    """Load model from checkpoint"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

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

    model = Layer2PretrainModel(model_cfg).to(device)

    # Fix: handle all checkpoint key formats
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    info = model.load_state_dict(state_dict, strict=False)
    if info.unexpected_keys:
        print(f"  Warning: unexpected keys: {info.unexpected_keys[:5]}...")
    if info.missing_keys:
        print(f"  Missing keys (random init): {info.missing_keys}")
    model.eval()

    if hasattr(model, "log_tau") and model.log_tau is not None:
        tau = torch.exp(model.log_tau).item()
        print(f"  Learned tau: {tau:.4f}")

    return model


def merge_accuracy_from_logits(logits, labels_10, scheme):
    """Merge via probability summing (more accurate)."""
    probs = torch.softmax(logits, dim=-1)
    n_classes = len(scheme)
    merged = torch.zeros(probs.shape[0], n_classes)
    for c, bins in enumerate(scheme):
        for b in bins:
            merged[:, c] += probs[:, b]
    merged_preds = merged.argmax(dim=-1).numpy()
    label_map = {}
    for c, bins in enumerate(scheme):
        for b in bins:
            label_map[b] = c
    mapped_labels = np.array([label_map[l] for l in labels_10])
    return float(np.mean(merged_preds == mapped_labels))


@torch.no_grad()
def evaluate(model: Layer2PretrainModel, dataloader: DataLoader, device: str = "cuda:0"):
    """Run evaluation and collect metrics"""
    model.eval()

    emb_correct_at = defaultdict(int)
    emb_total = 0
    cosine_sims = []

    yield_bin_logits_all = []
    yield_bin_trues = []
    yield_reg_trues = []

    amt_errors = defaultdict(list)
    task_counts = defaultdict(int)

    for batch in dataloader:
        out = model(batch)

        # --- Embedding ---
        if batch.emb_query_pos.numel() > 0:
            qp = batch.emb_query_pos.to(device)
            pos = batch.emb_pos.to(device)
            q = out["pred_emb"][qp[:, 0], qp[:, 1], :]
            q = F.normalize(q, p=2, dim=-1).float()
            pos_norm = F.normalize(pos, p=2, dim=-1).float()

            cos_sim = (q * pos_norm).sum(dim=-1)
            cosine_sims.extend(cos_sim.cpu().tolist())

            tau = 0.07
            if hasattr(model, "log_tau") and model.log_tau is not None:
                tau = torch.exp(model.log_tau).clamp(min=0.01, max=1.0).item()
            logits = (q @ pos_norm.t()) / tau
            targets = torch.arange(pos.size(0), device=device)
            n = pos.size(0)
            for k in [1, 5, 10]:
                if n >= k:
                    topk = logits.topk(min(k, n), dim=-1).indices
                    correct = (topk == targets.unsqueeze(-1)).any(dim=-1).sum().item()
                    emb_correct_at[k] += correct
            emb_total += n

        # --- Yield ---
        y_mask = batch.yield_pred_mask.to(device)
        idx = (y_mask > 0.5).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() > 0:
            y_bin = batch.yield_bin.to(device)[idx]
            y_reg = batch.yield_reg.to(device)[idx]
            pred_bin = out["pred_yield_bin"][idx].float()

            yield_bin_logits_all.append(pred_bin.cpu())
            yield_bin_trues.extend(y_bin.cpu().tolist())
            yield_reg_trues.extend(y_reg.cpu().tolist())

        # --- Amount ---
        if batch.amt_query_pos.numel() > 0:
            ap = batch.amt_query_pos.to(device)
            true_v = batch.amt_true.to(device)
            pred = out["pred_amt"][ap[:, 0], ap[:, 1], ap[:, 2]]
            channel_names = {0: "moles", 1: "mass", 2: "volume"}
            for i in range(ap.size(0)):
                ch_id = ap[i, 2].item()
                ch_name = channel_names.get(ch_id, f"ch{ch_id}")
                err = abs(pred[i].item() - true_v[i].item())
                amt_errors[ch_name].append(err)

        if batch.tasks is not None:
            task_names = {0: "forward/mask_product", 1: "yield_full", 2: "yield+mask_product",
                         3: "mask_condition", 4: "retro/mask_reactant"}
            for t in batch.tasks.tolist():
                task_counts[task_names.get(t, f"unknown_{t}")] += 1

    # === Compile ===
    results = {}

    # Embedding
    results["embedding"] = {}
    if emb_total > 0:
        for k in [1, 5, 10]:
            results["embedding"][f"top{k}_acc"] = emb_correct_at[k] / emb_total
        results["embedding"]["mean_cosine_sim"] = np.mean(cosine_sims) if cosine_sims else 0.0
        results["embedding"]["total_queries"] = emb_total

    # Yield (classification-based)
    results["yield"] = {}
    if yield_bin_trues:
        all_logits = torch.cat(yield_bin_logits_all, dim=0)
        bt = np.array(yield_bin_trues, dtype=int)
        bp = all_logits.argmax(dim=-1).numpy()
        rt = np.array(yield_reg_trues)

        # 10-class
        results["yield"]["bin_accuracy_10cls"] = float(np.mean(bt == bp))
        results["yield"]["relaxed_acc_10cls"] = float(np.mean(np.abs(bt - bp) <= 1))

        # 5-class / 3-class (from logits)
        results["yield"]["bin_accuracy_5cls"] = merge_accuracy_from_logits(all_logits, bt.tolist(), MERGE_5)
        results["yield"]["bin_accuracy_3cls"] = merge_accuracy_from_logits(all_logits, bt.tolist(), MERGE_3)

        # Softmax-expectation MAE (use classification head as regression)
        probs = torch.softmax(all_logits, dim=-1)
        soft_pred = (probs * BIN_CENTERS).sum(dim=-1).numpy()
        results["yield"]["softmax_mae"] = float(np.mean(np.abs(soft_pred - rt)))

        results["yield"]["n_samples"] = len(yield_bin_trues)

    # Amount
    results["amount"] = {}
    for ch, errors in amt_errors.items():
        results["amount"][f"{ch}_mae"] = float(np.mean(errors))
        results["amount"][f"{ch}_n_samples"] = len(errors)

    results["task_distribution"] = dict(task_counts)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Layer2 model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device=args.device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    print(f"Loading data from: {args.data}")
    dataset = Layer2JsonlIndexed(args.data, masking=True, masking_cfg=EvalMaskingConfig())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_layer2, num_workers=args.num_workers, pin_memory=True)
    print(f"  Samples: {len(dataset)}")

    print("Evaluating...")
    results = evaluate(model, loader, device=args.device)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    if results["embedding"]:
        print("\n--- Embedding Retrieval ---")
        for k, v in results["embedding"].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if results["yield"]:
        print("\n--- Yield Prediction ---")
        for k, v in results["yield"].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if results["amount"]:
        print("\n--- Amount Prediction ---")
        for k, v in results["amount"].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if results["task_distribution"]:
        print("\n--- Task Distribution ---")
        total = sum(results["task_distribution"].values())
        for task, count in sorted(results["task_distribution"].items()):
            print(f"  {task}: {count} ({100*count/total:.1f}%)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
