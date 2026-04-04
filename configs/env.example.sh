#!/bin/bash
# SciCore-Mol Environment Configuration
# Copy this file to configs/env.sh and fill in your paths

# Project root directory
export SCICORE_ROOT="/path/to/SciCore-Mol"

# Base model directory (e.g., Qwen3-8B, LlaSMol, etc.)
export MODEL_DIR="/path/to/models"

# Trained checkpoint directory
export CHECKPOINT_DIR="/path/to/checkpoints"

# Training and evaluation data directory
export DATA_DIR="/path/to/data"

# External dependencies (e.g., SMolInstruct scoring scripts)
export SMOLINSTRUCT_DIR="/path/to/SMolInstruct"

# GVP pretrained weights
export GVP_CHECKPOINT="/path/to/gvp_weights.pt"

# OpenAI API (for GPT baseline evaluation)
export OPENAI_API_KEY="your-api-key-here"
# export API_BASE="https://api.openai.com/v1"  # Uncomment to use custom API endpoint

# HuggingFace (uncomment to use mirror)
# export HF_ENDPOINT="https://huggingface.co"
