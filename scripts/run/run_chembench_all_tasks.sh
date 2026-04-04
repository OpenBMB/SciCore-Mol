#!/bin/bash
# ChemBench taskevaluationscript
# evaluation product, retro, yield task

set -euo pipefail

# ============================================
# pathconfig
# ============================================

# Qwen cpt+sft checkpoint
MOLAWARE_CKPT="${MOLAWARE_CKPT:-${CHECKPOINT_DIR:-/path/to/checkpoints}/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/checkpoint-4200}"

# Mol classifier
TOKEN_CLASSIFIER_PATH="${TOKEN_CLASSIFIER_PATH:-${CHECKPOINT_DIR:-/path/to/checkpoints}/gnn_classifier/qwen3_mlp_token_head.pt}"

# Base LLM (optional)
BASE_LLM_PATH="${BASE_LLM_PATH:-}"

# ============================================
# runconfig
# ============================================
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-cuda:0}"
DEVICE_MAP="${DEVICE_MAP:-}"
DTYPE="${DTYPE:-bf16}"

# ============================================
# generateparameter
# ============================================
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.9}"
REALTIME_MOL="${REALTIME_MOL:-1}"

# ============================================
# evaluationparameter
# ============================================
SPLIT="${SPLIT:-test}" # test dev
MAX_SAMPLES="${MAX_SAMPLES:-}" # empty

# ============================================
# Layer2 parameter
# ============================================
USE_LAYER2_PIPELINE="${USE_LAYER2_PIPELINE:-1}" # 1: Layer2 pipeline, 0: 
LAYER2_TASK_TYPE="${LAYER2_TASK_TYPE:-}" # optional: reaction_prediction, yield_prediction, product_prediction, reactant_prediction

# ============================================
# outputdirectory
# ============================================
OUTPUT_DIR="${OUTPUT_DIR:-${SCICORE_ROOT:-/path/to/scicore-mol}/eval_chembench_$(date +%Y%m%d_%H%M%S)}"

# ============================================
# scriptpath
# ============================================
SciCore-Mol_ROOT="${SciCore-Mol_ROOT:-${SCICORE_ROOT:-/path/to/scicore-mol}}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${SciCore-Mol_ROOT}/scripts/eval/eval_layer2_chembench.py}"

cd "$SciCore-Mol_ROOT"

# createoutputdirectorywritepermission
mkdir -p "$OUTPUT_DIR"
# ifdirectorycreatenewdirectory
if [ ! -w "$OUTPUT_DIR" ]; then
 echo "⚠️ warning: outputdirectorycreatedirectory..."
 OUTPUT_DIR="${SciCore-Mol_ROOT}/eval_chembench_$(date +%Y%m%d_%H%M%S)"
 mkdir -p "$OUTPUT_DIR"
 echo " usedirectory: $OUTPUT_DIR"
fi

echo "============================================"
echo "ChemBench taskevaluation"
echo "============================================"
echo "Split: $SPLIT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Device: $DEVICE"
echo "Dtype: $DTYPE"
echo "MolAware Checkpoint: $MOLAWARE_CKPT"
echo "Token Classifier: $TOKEN_CLASSIFIER_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Max Samples: ${MAX_SAMPLES:-<all>}"
echo "Gen Params: max_tokens=$MAX_NEW_TOKENS temp=$TEMPERATURE top_p=$TOP_P realtime_mol=$REALTIME_MOL"
echo "Layer2 Pipeline: ${USE_LAYER2_PIPELINE:-0} (${LAYER2_TASK_TYPE:-auto})"
echo "============================================"

# buildparameter
BASE_ARGS=(
 --molaware_ckpt "$MOLAWARE_CKPT"
 --token_classifier_path "$TOKEN_CLASSIFIER_PATH"
 --device "$DEVICE"
 --dtype "$DTYPE"
 --split "$SPLIT"
 --max_new_tokens "$MAX_NEW_TOKENS"
 --temperature "$TEMPERATURE"
 --top_p "$TOP_P"
 --realtime_mol "$REALTIME_MOL"
)

# optionalparameter
[[ -n "$BASE_LLM_PATH" ]] && BASE_ARGS+=( --base_llm_path "$BASE_LLM_PATH" )
[[ -n "$DEVICE_MAP" ]] && BASE_ARGS+=( --device_map "$DEVICE_MAP" )
[[ -n "$MAX_SAMPLES" ]] && BASE_ARGS+=( --max_samples "$MAX_SAMPLES" )

# Layer2 parameter
BASE_ARGS+=( --use_layer2_pipeline "$USE_LAYER2_PIPELINE" )
[[ -n "$LAYER2_TASK_TYPE" ]] && BASE_ARGS+=( --layer2_task_type "$LAYER2_TASK_TYPE" )

# tasklist
TASKS=("product" "retro" "yield")

# evaluationeachtask
for task in "${TASKS[@]}"; do
 echo ""
 echo "============================================"
 echo "evaluationtask: $task ($SPLIT split)"
 echo ": $(date)"
 echo "============================================"
 
 ARGS=("${BASE_ARGS[@]}")
 ARGS+=( --task "$task" )
 ARGS+=( --out_dir "$OUTPUT_DIR" )
 
 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python "$EVAL_SCRIPT" "${ARGS[@]}"
 
 echo "✅ task $task complete"
done

echo ""
echo "============================================"
echo "alltaskevaluationcomplete"
echo "============================================"
echo ""
echo "outputfile:"
echo " - $OUTPUT_DIR/pred_product.jsonl"
echo " - $OUTPUT_DIR/pred_retro.jsonl"
echo " - $OUTPUT_DIR/pred_yield.jsonl"
echo ""
echo "result:"
echo " - $OUTPUT_DIR/chembench4k_product_${SPLIT}_predictions.jsonl"
echo " - $OUTPUT_DIR/chembench4k_retro_${SPLIT}_predictions.jsonl"
echo " - $OUTPUT_DIR/chembench4k_yield_${SPLIT}_predictions.jsonl"
echo ""
echo "file:"
echo " - $OUTPUT_DIR/chembench4k_product_${SPLIT}_summary.json"
echo " - $OUTPUT_DIR/chembench4k_retro_${SPLIT}_summary.json"
echo " - $OUTPUT_DIR/chembench4k_yield_${SPLIT}_summary.json"
echo ""
