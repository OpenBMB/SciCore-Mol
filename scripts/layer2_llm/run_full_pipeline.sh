#!/bin/bash
# Layer2-LLM pipelinegeneratedata -> training -> evaluation
# usemethod: bash scripts/layer2_llm/run_full_pipeline.sh

set -e # errordefinevariablecheck conda 

# ============================================
# configparameter
# ============================================

# directory
SciCore-Mol_ROOT="${SciCore-Mol_ROOT:-${SCICORE_ROOT:-/path/to/scicore-mol}}"
cd "$SciCore-Mol_ROOT"

# environmentconfig
CONDA_ENV="${CONDA_ENV:-llam3.2}"
# CUDA_VISIBLE_DEVICES fortraining GPU
# datageneratestageset GPU
TRAIN_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
NUM_GPUS="${NUM_GPUS:-4}"

# ============================================
# stage1: generatetrainingdata
# ============================================

# input datause ChemBench
USE_CHEMBENCH="${USE_CHEMBENCH:-1}" # 1: use ChemBench, 0: usefile
CHEMBENCH_TASK="${CHEMBENCH_TASK:-product}" # product, retro, yield
CHEMBENCH_SPLIT="${CHEMBENCH_SPLIT:-dev}" # dev, test (ChemBench trainuse dev trainingdata)
TRAIN_DATA_INPUT="${TRAIN_DATA_INPUT:-}" # ifusefileotherwiseuse ChemBench
TRAIN_DATA_OUTPUT="${TRAIN_DATA_OUTPUT:-${SciCore-Mol_ROOT}/scripts/layer2_llm/data/training_data_${CHEMBENCH_TASK}_${CHEMBENCH_SPLIT}.jsonl}"

# modelconfigforgeneratedata
GEN_CONFIG="${GEN_CONFIG:-${SciCore-Mol_ROOT}/configs/qwen3_sft_epoch2_3.yaml}"
# defaultuse GPU 6if GPU 6 7datagenerate GPU 6
GEN_DEVICE="${GEN_DEVICE:-cuda:6}"
GEN_TASK_TYPE="${GEN_TASK_TYPE:-reaction_prediction}"

# ============================================
# stage2: training LLM
# ============================================

# trainingconfig
TRAIN_CONFIG="${TRAIN_CONFIG:-${SciCore-Mol_ROOT}/configs/qwen3_sft_epoch2_3.yaml}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${CHECKPOINT_DIR:-/path/to/checkpoints}/qwen3_8b_layer2_llm_$(date +%Y%m%d_%H%M%S)}"
TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-29500}"

# ============================================
# stage3: evaluation ChemBench
# ============================================

# evaluationconfig
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${SciCore-Mol_ROOT}/eval_chembench_layer2_llm_$(date +%Y%m%d_%H%M%S)}"
# defaultuse GPU 7if GPU 6 7evaluation GPU 7
EVAL_DEVICE="${EVAL_DEVICE:-cuda:7}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
TOKEN_CLASSIFIER_PATH="${TOKEN_CLASSIFIER_PATH:-${CHECKPOINT_DIR:-/path/to/checkpoints}/gnn_classifier/qwen3_mlp_token_head.pt}"

# ============================================
# scriptpath
# ============================================

GENERATE_SCRIPT="${SciCore-Mol_ROOT}/scripts/layer2_llm/generate_training_data.py"
TRAIN_SCRIPT="${SciCore-Mol_ROOT}/train_sft.py"
EVAL_SCRIPT="${SciCore-Mol_ROOT}/scripts/eval/eval_layer2_chembench.py"

# ============================================
# startexecute
# ============================================

echo "============================================"
echo "Layer2-LLM pipeline"
echo "============================================"
echo "directory: $SciCore-Mol_ROOT"
echo "training GPU: $TRAIN_CUDA_VISIBLE_DEVICES"
echo "trainingoutput: $TRAIN_OUTPUT_DIR"
echo "evaluationoutput: $EVAL_OUTPUT_DIR"
echo "============================================"

# environment
source ${SCICORE_ROOT:-/path/to/scicore-mol}/.venv/bin/activate

# ============================================
# stage1: generatetrainingdata
# ============================================

echo ""
echo "============================================"
echo "stage1: generatetrainingdata"
echo "============================================"
if [ -n "$TRAIN_DATA_INPUT" ]; then
 echo "input: $TRAIN_DATA_INPUT"
else
 echo "input: ChemBench ($CHEMBENCH_TASK/$CHEMBENCH_SPLIT)"
fi
echo "output: $TRAIN_DATA_OUTPUT"
echo "config: $GEN_CONFIG"
echo "device: $GEN_DEVICE"
echo ""

# createoutputdirectory
mkdir -p "$(dirname "$TRAIN_DATA_OUTPUT")"

# generatetrainingdata
# processdevicemappingif GEN_DEVICE cuda:Xset CUDA_VISIBLE_DEVICES=Xuse cuda:0
# NOTEset CUDA_VISIBLE_DEVICES GPU mapping GPU 0,1,2...
# ifset CUDA_VISIBLE_DEVICES=6 GPU 6 GPU 0
GEN_DEVICE_MAPPED="$GEN_DEVICE"
if [[ "$GEN_DEVICE" == cuda:* ]]; then
 GPU_ID=$(echo "$GEN_DEVICE" | sed 's/cuda://')
 export CUDA_VISIBLE_DEVICES="$GPU_ID"
 GEN_DEVICE_MAPPED="cuda:0"
 echo "📌 datageneratedevicemapping: GPU $GPU_ID -> GPU 0"
fi

GEN_ARGS=(
 --output "$TRAIN_DATA_OUTPUT"
 --config "$GEN_CONFIG"
 --task_type "$GEN_TASK_TYPE"
 --device "$GEN_DEVICE_MAPPED"
)

# ifinputfileusefileotherwiseuse ChemBench
if [ -n "$TRAIN_DATA_INPUT" ] && [ -f "$TRAIN_DATA_INPUT" ]; then
 echo "📂 useinputfile: $TRAIN_DATA_INPUT"
 GEN_ARGS+=( --input "$TRAIN_DATA_INPUT" )
else
 echo "📂 use ChemBench data: task=$CHEMBENCH_TASK, split=$CHEMBENCH_SPLIT"
 GEN_ARGS+=( --use_chembench )
 GEN_ARGS+=( --chembench_task "$CHEMBENCH_TASK" )
 GEN_ARGS+=( --chembench_split "$CHEMBENCH_SPLIT" )
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python "$GENERATE_SCRIPT" "${GEN_ARGS[@]}"

if [ ! -f "$TRAIN_DATA_OUTPUT" ]; then
 echo "❌ trainingdatageneratefail"
 exit 1
fi

DATA_COUNT=$(wc -l < "$TRAIN_DATA_OUTPUT")
echo "✅ trainingdatageneratecomplete: $DATA_COUNT "

# ============================================
# stage2: training LLM
# ============================================

echo ""
echo "============================================"
echo "stage2: training LLM"
echo "============================================"
echo "trainingdata: $TRAIN_DATA_OUTPUT"
echo "configfile: $TRAIN_CONFIG"
echo "outputdirectory: $TRAIN_OUTPUT_DIR"
echo "GPUcount: $NUM_GPUS"
echo ""

# checktrainingdata
if [ ! -f "$TRAIN_DATA_OUTPUT" ]; then
 echo "❌ trainingdata: $TRAIN_DATA_OUTPUT"
 exit 1
fi

# checkconfigfile
if [ ! -f "$TRAIN_CONFIG" ]; then
 echo "❌ trainingconfigfile: $TRAIN_CONFIG"
 exit 1
fi

# createconfigfileupdatedatapath
TEMP_CONFIG="${TRAIN_CONFIG%.yaml}_layer2_temp.yaml"
cp "$TRAIN_CONFIG" "$TEMP_CONFIG"

# updatedatapathuse Python sed
python3 << EOF
import yaml
import sys

with open("$TEMP_CONFIG", 'r') as f:
 config = yaml.safe_load(f)

# updatedatapath
if 'data' not in config:
 config['data'] = {}
config['data']['dataset_path'] = "$TRAIN_DATA_OUTPUT"

with open("$TEMP_CONFIG", 'w') as f:
 yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print(f"✅ updateconfigfile: $TEMP_CONFIG")
print(f" datapath: $TRAIN_DATA_OUTPUT")
EOF

# useconfigfile
ACTUAL_TRAIN_CONFIG="$TEMP_CONFIG"

# createoutputdirectory
mkdir -p "$TRAIN_OUTPUT_DIR"

# training LLM
echo "starttraining..."
CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" torchrun \
 --master_port="$TRAIN_MASTER_PORT" \
 --nproc_per_node="$NUM_GPUS" \
 "$TRAIN_SCRIPT" \
 --config "$ACTUAL_TRAIN_CONFIG" \
 --output_dir "$TRAIN_OUTPUT_DIR"

# configfile
if [ -f "$TEMP_CONFIG" ]; then
 rm "$TEMP_CONFIG"
 echo "✅ configfile"
fi

# checktrainingwhethersuccessnew checkpoint
# waitfilesync
sleep 2

LATEST_CKPT=$(find "$TRAIN_OUTPUT_DIR" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
if [ -z "$LATEST_CKPT" ]; then
 echo "⚠️ warning: training checkpointtrainingfailtraining"
 echo " outputdirectory: $TRAIN_OUTPUT_DIR"
 echo ""
 read -p "whetheruseoutputdirectory checkpoint continueevaluation(y/n) " -n 1 -r
 echo
 if [[ ! $REPLY =~ ^[Yy]$ ]]; then
 echo "❌ cancelevaluation"
 exit 1
 fi
 # useoutputdirectory checkpoint
 MOLAWARE_CKPT="$TRAIN_OUTPUT_DIR"
 echo " useoutputdirectory: $MOLAWARE_CKPT"
else
 # usenew checkpoint
 MOLAWARE_CKPT="$LATEST_CKPT"
 echo "✅ trainingcomplete checkpoint: $MOLAWARE_CKPT"
 
 # checkwhether llm directoryconfiguse
 if [ -d "$MOLAWARE_CKPT/llm" ]; then
 MOLAWARE_CKPT="$MOLAWARE_CKPT/llm"
 echo " use llm directory: $MOLAWARE_CKPT"
 fi
fi

# ============================================
# stage3: evaluation ChemBench
# ============================================

echo ""
echo "============================================"
echo "stage3: evaluation ChemBench"
echo "============================================"
echo "Checkpoint: $MOLAWARE_CKPT"
echo "outputdirectory: $EVAL_OUTPUT_DIR"
echo "evaluation: $EVAL_SPLIT"
echo ""

# createoutputdirectory
mkdir -p "$EVAL_OUTPUT_DIR"

# evaluationtask
TASKS=("product" "retro" "yield")

for task in "${TASKS[@]}"; do
 echo ""
 echo "--------------------------------------------"
 echo "evaluationtask: $task"
 echo "--------------------------------------------"
 
 # processdevicemappingif EVAL_DEVICE cuda:Xset CUDA_VISIBLE_DEVICES=Xuse cuda:0
 EVAL_CUDA_VISIBLE_DEVICES=""
 EVAL_DEVICE_MAPPED="$EVAL_DEVICE"
 if [[ "$EVAL_DEVICE" == cuda:* ]]; then
 EVAL_GPU_ID=$(echo "$EVAL_DEVICE" | sed 's/cuda://')
 EVAL_CUDA_VISIBLE_DEVICES="$EVAL_GPU_ID"
 EVAL_DEVICE_MAPPED="cuda:0"
 echo "📌 evaluationdevicemapping: GPU $EVAL_GPU_ID -> GPU 0"
 fi
 
 CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES" python "$EVAL_SCRIPT" \
 --task "$task" \
 --split "$EVAL_SPLIT" \
 --molaware_ckpt "$MOLAWARE_CKPT" \
 --token_classifier_path "$TOKEN_CLASSIFIER_PATH" \
 --device "$EVAL_DEVICE_MAPPED" \
 --dtype bf16 \
 --out_dir "$EVAL_OUTPUT_DIR" \
 --use_layer2_pipeline 1 \
 --max_new_tokens 256 \
 --temperature 0.2 \
 --top_p 0.9
 
 echo "✅ task $task evaluationcomplete"
done

# ============================================
# total
# ============================================

echo ""
echo "============================================"
echo "✅ pipelineexecutecomplete"
echo "============================================"
echo ""
echo "trainingdata: $TRAIN_DATA_OUTPUT"
echo "trainingoutput: $TRAIN_OUTPUT_DIR"
echo "evaluationoutput: $EVAL_OUTPUT_DIR"
echo ""
echo "evaluationresultfile:"
echo " - $EVAL_OUTPUT_DIR/pred_product.jsonl"
echo " - $EVAL_OUTPUT_DIR/pred_retro.jsonl"
echo " - $EVAL_OUTPUT_DIR/pred_yield.jsonl"
echo ""
echo "result:"
for task in "${TASKS[@]}"; do
 echo " - $EVAL_OUTPUT_DIR/chembench4k_${task}_${EVAL_SPLIT}_predictions.jsonl"
 echo " - $EVAL_OUTPUT_DIR/chembench4k_${task}_${EVAL_SPLIT}_summary.json"
done
echo ""
