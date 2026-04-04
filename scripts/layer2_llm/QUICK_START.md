# Layer2-LLM 

## 

```bash
cd ${SCICORE_ROOT:-/path/to/scicore-mol}

# 
bash scripts/layer2_llm/run_full_pipeline.sh
```

## 

```bash
# 
export TRAIN_DATA_INPUT="/path/to/queries.jsonl"
export TRAIN_CONFIG="/path/to/config.yaml"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NUM_GPUS=4

# 
bash scripts/layer2_llm/run_full_pipeline.sh
```

## 



1. **** - Layer2 pipeline 
2. ** LLM** - 
3. ** ChemBench** - product, retro, yield 

## 

- ****: `scripts/layer2_llm/data/training_data.jsonl`
- ****: `${CHECKPOINT_DIR:-/path/to/checkpoints}/qwen3_8b_layer2_llm_YYYYMMDD_HHMMSS/`
- ****: `eval_chembench_layer2_llm_YYYYMMDD_HHMMSS/`

## 

```bash
# 
cat eval_chembench_layer2_llm_*/pred_product.jsonl | head -5
cat eval_chembench_layer2_llm_*/pred_retro.jsonl | head -5
cat eval_chembench_layer2_llm_*/pred_yield.jsonl | head -5

# 
cat eval_chembench_layer2_llm_*/chembench4k_*_test_summary.json | grep acc
```

## 

: `../../LAYER2_LLM_TRAINING_GUIDE.md`
