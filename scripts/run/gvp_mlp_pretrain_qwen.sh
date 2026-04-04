#!/bin/bash
# training GNN MLP adapterQwen 30k data

# condaenvironment
source ${CONDA_PREFIX:-/path/to/conda}/etc/profile.d/conda.sh
conda activate llam3.2

# directory
cd ${SCICORE_ROOT:-/path/to/scicore-mol}

# runtraining
python scripts/train/gvp_mlp_pretrain.py \
 --data ${DATA_DIR:-/path/to/data}/MSMLM/data/traindata/chatmol/chatmol_gnn_qwen_30k.pkl \
 --outdir ${DATA_DIR:-/path/to/data}/MSMLM/model/gnn_mlp_qwen \
 --gnn-class modules.gnn:GVPEncoder \
 --gnn-ckpt ${GVP_CHECKPOINT:-/path/to/gvp_weights.pt} \
 --gnn-batch-size 128 \
 --hidden-dim 1536 \
 --num-layers 2 \
 --epochs 100 \
 --batch-size 256 \
 --lr 1e-5 \
 --alpha 0.5 \
 --weight-decay 0.0 \
 --scheduler plateau \
 --grad-clip 1.0 \
 --target-normalize none \
 --val-ratio 0.05 \
 --seed 42 \
 --use-cache \
 --tensorboard \
 --bf16 \
 --amp 

