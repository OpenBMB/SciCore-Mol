#!/bin/bash
# runscoringscriptusefix

# predictiondirectory
DIRS=(
 "${SCICORE_ROOT:-/path/to/scicore-mol}/test_output_eval_qwen_v24-20251216-225348_checkpoint-800"
 "${SCICORE_ROOT:-/path/to/scicore-mol}/test_output_eval_qwen_v6-20251217-220917_checkpoint-5938"
 "${SCICORE_ROOT:-/path/to/scicore-mol}/test_output_eval_qwen_GNN_nofreeze_checkpoint-39"
 "${SCICORE_ROOT:-/path/to/scicore-mol}/test_output_eval_qwen_LLM_nofreeze_checkpoint-400"
)

SCRIPT_DIR="${SMOLINSTRUCT_DIR:-/path/to/SMolInstruct}"

for DIR in "${DIRS[@]}"; do
 echo "========================================="
 echo "Scoring: $DIR"
 echo "========================================="
 
 python "${SCRIPT_DIR}/score_smolinstruct.py" \
 --prediction_dir "$DIR" \
 --save_json "${DIR}/scored_results.json"
 
 echo ""
done

echo "All scoring completed!"

