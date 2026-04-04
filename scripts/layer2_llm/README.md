# Layer2 + LLM 

 Layer2 LLM 

## 

- `generate_training_data.py`: LLM + Layer2 
- `train_layer2_llm.sh`: LLM + Layer2 
- `quick_test.sh`: 

## 

### 1. 

```bash
python generate_training_data.py \
 --input /path/to/queries.jsonl \
 --output /path/to/training_data.jsonl \
 --config /path/to/model_config.yaml \
 --task_type "reaction_prediction"
```

### 2. 

```bash
bash train_layer2_llm.sh
```

### 3. 

```bash
bash quick_test.sh
```

## 

### queries.jsonl

```json
{"input": "query text"}
```

### training_data.jsonl

```json
{
 "input": " query",
 "intermediate": " JSON ",
 "molecules_info": [
 {
 "smiles": "CCO",
 "role": "REACTANT",
 "amount_info": {...}
 }
 ],
 "layer2_info": {
 "yield_bin": 5,
 "yield_reg": 0.75,
 "embedding_shape": [1024]
 },
 "output": " LLM "
}
```

## 

- `../../LAYER2_GUIDE.md`
- `../../LAYER2_TEST_INSTRUCTIONS.md`
