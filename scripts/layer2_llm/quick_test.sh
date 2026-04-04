#!/bin/bash
# Layer2 testscript

set -e

PROJECT_ROOT="${SCICORE_ROOT:-/path/to/scicore-mol}"
cd ${PROJECT_ROOT}

echo "============================================"
echo "Layer2 testpipeline"
echo "============================================"

# 1. checkdata
echo ""
echo "1. checkdata..."
LAYER2_DATA="${SCICORE_ROOT:-/path/to/scicore-mol}/Layer2/data"
if [ ! -f "${LAYER2_DATA}/ord_layer2/layer2_test.jsonl" ]; then
 echo "⚠️ testdata: ${LAYER2_DATA}/ord_layer2/layer2_test.jsonl"
 echo " datadownload"
else
 echo "✅ datacheckvia"
fi

# 2. checkconfigfile
echo ""
echo "2. checkconfigfile..."
if [ ! -f "modules/layer2_component/layer2_config.yaml" ]; then
 echo "❌ Layer2 configfile"
 exit 1
fi
if [ ! -f "scripts/layer2/layer2_train_config.yaml" ]; then
 echo "❌ Layer2 trainingconfigfile"
 exit 1
fi
echo "✅ configfilecheckvia"

# 3. test Python 
echo ""
echo "3. test Python ..."
python -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')

try:
 from modules.layer2_component.Layer2Inferer import Layer2Inferer
 print('✅ Layer2Inferer success')
except Exception as e:
 print(f'❌ Layer2Inferer fail: {e}')
 sys.exit(1)

try:
 from sft_tester import MolAwareGenerator2
 print('✅ MolAwareGenerator2 success')
except Exception as e:
 print(f'❌ MolAwareGenerator2 fail: {e}')
 sys.exit(1)
"

# 4. test JSON parse
echo ""
echo "4. test JSON parse..."
python -c "
import json
import re

# test JSON format
test_json = '''
{
 \"molecules\": [
 {
 \"smiles\": \"CCO\",
 \"role\": \"REACTANT\",
 \"amount_info\": {
 \"moles\": 1.0
 }
 }
 ]
}
'''

try:
 parsed = json.loads(test_json)
 if 'molecules' in parsed:
 print('✅ JSON formatvalidatevia')
 else:
 print('❌ JSON formatvalidatefail')
 sys.exit(1)
except Exception as e:
 print(f'❌ JSON parsefail: {e}')
 sys.exit(1)
"

# 5. checkdependency
echo ""
echo "5. checkdependency..."
python -c "
try:
 import json_repair
 print('✅ json-repair install')
except ImportError:
 print('⚠️ json-repair installinstall: pip install json-repair')

try:
 import torch
 print(f'✅ torch install: {torch.__version__}')
except ImportError:
 print('❌ torch install')
 sys.exit(1)
"

echo ""
echo "============================================"
echo "✅ checkcomplete"
echo "============================================"
echo ""
echo ""
echo "1. run Layer2 training: bash scripts/layer2/train_layer2.py"
echo "2. run Layer2 evaluation: bash scripts/run/run_eval_layer2_testset.sh"
echo "3. generatetrainingdata: python scripts/layer2_llm/generate_training_data.py"
echo "4. training LLM+Layer2: bash scripts/layer2_llm/train_layer2_llm.sh"
echo ""
echo ": LAYER2_TEST_INSTRUCTIONS.md"
