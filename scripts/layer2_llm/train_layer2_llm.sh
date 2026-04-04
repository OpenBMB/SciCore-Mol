#!/bin/bash
# stagetraining LLMusestagegeneratedata

# defaultconfig
CODE_DIR="${SCICORE_ROOT:-/path/to/scicore-mol}"
CONFIG_FILE="${CODE_DIR}/configs/qwen3_sft_epoch2_layer2.yaml"
OUTPUT_DIR="${CHECKPOINT_DIR:-/path/to/checkpoints}/qwen3_8b_layer2_llm"
TRAIN_DATA="${CODE_DIR}/scripts/layer2_llm/data/training_data.jsonl"

# allowviaparameter
if [ "$1" != "" ]; then
 CONFIG_FILE="$1"
fi
if [ "$2" != "" ]; then
 OUTPUT_DIR="$2"
fi

# environment
source ${CONDA_PREFIX:-/path/to/conda}/etc/profile.d/conda.sh
conda activate llam3.2

cd ${CODE_DIR}

# training
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
 --master_port=29500 \
 --nproc_per_node=4 \
 train_sft.py \
 --config ${CONFIG_FILE} \
 --output_dir ${OUTPUT_DIR}

echo "✅ trainingcomplete"
