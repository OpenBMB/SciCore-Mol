#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# LDMol trainingtestscript
#
# :
# # single GPUtraining
# bash test_trainer.sh
#
# # multi-GPUtraining
# GPUS=0,1,2,3 NPROC=4 bash test_trainer.sh
#
# # defineparameter
# DATA_PATH=./data/my_train.txt EPOCHS=50 bash test_trainer.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PWD}"

# ----- configparameter -----
: "${GPUS:=0}" # GPU devicemulti-GPU
: "${NPROC:=1}" # GPU count
: "${MASTER_PORT:=29500}" # DDP master port

: "${DATA_PATH:=./data/chatmol/train.txt}" # trainingdatapath
: "${TEXT_ENCODER_PATH:=${DATA_DIR:-/path/to/data}/base_model/qwen3_8b}" # Text Encoder path
: "${VAE_CKPT_PATH:=${CHECKPOINT_DIR:-/path/to/checkpoints}/diffusion_pretrained/official/checkpoint_autoencoder.ckpt}" # VAE weight
: "${LDMOL_CKPT_PATH:=}" # DiT trainingweightoptional

: "${EPOCHS:=100}" # training
: "${GLOBAL_BATCH_SIZE:=64}" # global batch size
: "${DESCRIPTION_LENGTH:=256}" # maxlength
: "${RESULTS_DIR:=./training_output}" # outputdirectory

: "${LOG_EVERY:=100}" # log
: "${CKPT_EVERY:=5000}" # save
: "${SEED:=0}" # random

# ----- executetraining -----
echo "=============================================="
echo "LDMol Training"
echo "=============================================="
echo "GPUs: ${GPUS} (${NPROC} processes)"
echo "Data: ${DATA_PATH}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${GLOBAL_BATCH_SIZE}"
echo "Output: ${RESULTS_DIR}"
echo "=============================================="

CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
 --nproc_per_node="${NPROC}" \
 --master_port="${MASTER_PORT}" \
 -m ldmol_component.LDMolTrainer \
 --data_path "${DATA_PATH}" \
 --text_encoder_path "${TEXT_ENCODER_PATH}" \
 --vae_ckpt_path "${VAE_CKPT_PATH}" \
 ${LDMOL_CKPT_PATH:+--ldmol_ckpt_path "${LDMOL_CKPT_PATH}"} \
 --epochs "${EPOCHS}" \
 --global_batch_size "${GLOBAL_BATCH_SIZE}" \
 --description_length "${DESCRIPTION_LENGTH}" \
 --results_dir "${RESULTS_DIR}" \
 --log_every "${LOG_EVERY}" \
 --ckpt_every "${CKPT_EVERY}" \
 --global_seed "${SEED}"
