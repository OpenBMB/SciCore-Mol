#!/bin/bash
set -euo pipefail

# Align CLI style with run_eval_moleculenet_qwen.sh, but use sft_tester pipeline.

PY="${SCICORE_ROOT:?SCICORE_ROOT not set}/.venv/bin/python"
CKPT="${CHECKPOINT_DIR:?CHECKPOINT_DIR not set}"
OUT="${SCICORE_ROOT}/eval/eval_moleculenet_cpt_sft_tester"
MAX_SAMPLES=100

# classification
for ds in BBBP Tox21 ClinTox HIV BACE
do
  echo "Running $ds ..."
  "$PY" eval_moleculenet_sft_tester.py \
    --ckpt_dir "$CKPT" \
    --dataset "$ds" \
    --split test \
    --n_shot 5 \
    --dtype bf16 \
    --max_new_tokens 128 \
    --output_dir "$OUT" \
    --max_samples "$MAX_SAMPLES"
done

# regression
for ds in ESOL FreeSolv Lipo QM9
do
  echo "Running $ds ..."
  "$PY" eval_moleculenet_sft_tester.py \
    --ckpt_dir "$CKPT" \
    --dataset "$ds" \
    --split test \
    --n_shot 5 \
    --dtype bf16 \
    --max_new_tokens 128 \
    --output_dir "$OUT" \
    --max_samples "$MAX_SAMPLES"
done

