#!/bin/bash
# parallelevaluationscript - viaconfiglist
# supportsmodel pathoutputfewshotGPUtaskchooseoptionalparameter
#
# usemethod
# 1. TASK_LIST evaluationtask
# 2. runscript: bash eval_smol_task_list.sh
#
# ==================== optionaltasklist ====================
# alloptionalevaluationtaskvia include_tasks parameter
# - molecule_generation # moleculegenerate
# - molecule_captioning # moleculedescription
# - name_conversion-i2f # IUPACmolecule
# - name_conversion-i2s # IUPACSMILES
# - name_conversion-s2f # SMILESmolecule
# - name_conversion-s2i # SMILESIUPAC
# - forward_synthesis # 
# - retrosynthesis # 
# - property_prediction-bbbp # BBBPprediction
# - property_prediction-clintox # ClinToxprediction
# - property_prediction-esol # ESOLprediction
# - property_prediction-hiv # HIVprediction
# - property_prediction-lipo # Lipoprediction
# - property_prediction-sider # SIDERprediction
#
# if include_tasksrunalltask
#
# ==================== taskconfigformat ====================
# eachtask | format
# model_path|output_name|fewshot|gpu|tasks|extra_args
#
# description
# - model_path: model pathrequired
# - output_name: outputoptionalemptymodel pathgenerate
# - fewshot: true/falserequired
# - gpu: GPU IDrequired 6
# - tasks: evaluationtasklistoptional molecule_generation,forward_synthesisemptyrunalltask
# - extra_args: optionalparameteroptionalformatkey1=value1,key2=value2
#
# ==================== useexample ====================
# example1evaluation molecule_generation task
# declare -a TASK_LIST=(
# "${CHECKPOINT_DIR:-/path/to/checkpoints}/model1||true|6|molecule_generation|"
# )
#
# example2evaluationtask
# declare -a TASK_LIST=(
# "${CHECKPOINT_DIR:-/path/to/checkpoints}/model1||true|6|molecule_generation,forward_synthesis|"
# )
#
# example3setfewshot/no fewshot × n-gramdefault/
# declare -a TASK_LIST=(
# # fewshot + n-gramdefault3
# "${CHECKPOINT_DIR:-/path/to/checkpoints}/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/checkpoint-4200||true|6|molecule_generation|"
# # fewshot + n-gramset0
# "${CHECKPOINT_DIR:-/path/to/checkpoints}/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/checkpoint-4200||true|7|molecule_generation|no_repeat_ngram_size=0"
# # no fewshot + n-gramdefault3
# "${CHECKPOINT_DIR:-/path/to/checkpoints}/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/checkpoint-4200||false|6|molecule_generation|"
# # no fewshot + n-gramset0
# "${CHECKPOINT_DIR:-/path/to/checkpoints}/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/checkpoint-4200||false|7|molecule_generation|no_repeat_ngram_size=0"
# )
#
# example4parameter
# declare -a TASK_LIST=(
# "${CHECKPOINT_DIR:-/path/to/checkpoints}/model1||true|6|molecule_generation|batch_size=8,data_limit=50"
# )
#
# ==================== supportsparameterextra_args ====================
# - batch_size: batchsizedefault: 16
# - data_limit: datalimitdefault: 100
# - max_new_tokens: maxgeneratetokendefault: 512
# - temperature: default: 0.2
# - top_p: top_psampledefault: 0.9
# - repetition_penalty: default: 1.06
# - no_repeat_ngram_size: n-gramlimitdefault: 3set0n-gram
# - realtime_mol: moleculeprocessdefault: 0
# - few_shot: fewshotcountdefault: 2fewshot=truevalid
# - prompt_style: prompt/hintdefault: strict

cd ${SCICORE_ROOT:-/path/to/scicore-mol}

# ==================== environmentcheck ====================
# checkbashversionwait -n needs bash 4.3+
BASH_VERSION_CHECK=$(bash --version | head -n1 | grep -oE '[0-9]+\.[0-9]+' | head -n1)
BASH_MAJOR=$(echo "$BASH_VERSION_CHECK" | cut -d. -f1)
BASH_MINOR=$(echo "$BASH_VERSION_CHECK" | cut -d. -f2)

if [ "$BASH_MAJOR" -lt 4 ] || ([ "$BASH_MAJOR" -eq 4 ] && [ "$BASH_MINOR" -lt 3 ]); then
 echo "⚠️ warning: bash version $BASH_VERSION_CHECK supports wait -nneeds 4.3+"
 echo " ifparallelfailbashusemode"
fi

# ==================== config ====================

# use SMolInstruct testdata
SMOLINSTRUCT_DIR="${SMOLINSTRUCT_DIR:-/path/to/SMolInstruct}"
RAW_DATA_DIR="${SMOLINSTRUCT_DIR}/constructed_test"
TEMPLATE_DIR="${SMOLINSTRUCT_DIR}/data/template/instruction_tuning"
DEV_DATA_DIR="${SMOLINSTRUCT_DIR}/data/constructed_dev"
TOKEN_CLS_PATH="${CHECKPOINT_DIR:-/path/to/checkpoints}/qwen3_mlp_token_head.pt"
MODEL_DIR="${DATA_DIR:-/path/to/data}/base_model"

# outputdirectoryviaenvironmentvariable
# SciCore-Mol_ROOT/eval_results/results directory results
SciCore-Mol_ROOT="${SciCore-Mol_ROOT:-${SCICORE_ROOT:-/path/to/scicore-mol}}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${SciCore-Mol_ROOT}/eval_results/results/smol_eval_$(date +%Y%m%d_%H%M%S)}"

# defaultevaluationparametercantaskconfig
DEFAULT_MAX_NEW_TOKENS=512
DEFAULT_TEMPERATURE=0.2
DEFAULT_TOP_P=0.9
DEFAULT_REPETITION_PENALTY=1.06
DEFAULT_NO_REPEAT_NGRAM_SIZE=3
DEFAULT_DATA_LIMIT=100
DEFAULT_FEW_SHOT=2
DEFAULT_FEW_SHOT_SEED=42
DEFAULT_PROMPT_STYLE="strict"
DEFAULT_BATCH_SIZE=16
DEFAULT_REALTIME_MOL=1

# ==================== taskconfiglist ====================
# formatmodel_path|output_name|fewshot|gpu|tasks|extra_args
# NOTEtasks emptyrunalltask

declare -a TASK_LIST=(
 # examplesetfewshot/no fewshot × n-gramdefault/evaluation molecule_generation
 "${CHECKPOINT_DIR:-/path/to/checkpoints}/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/name_conversion/checkpoint-268|qwen3_8b_cpt_sft_gvp_name_conversion_fewshot_ngram0|true|0||no_repeat_ngram_size=0"
 "${CHECKPOINT_DIR:-/path/to/checkpoints}/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/name_conversion/checkpoint-268|qwen3_8b_cpt_sft_gvp_name_conversion_nofewshot_ngram0|false|1||no_repeat_ngram_size=0"
 "${CHECKPOINT_DIR:-/path/to/checkpoints}/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/name_conversion/checkpoint-268|qwen3_8b_cpt_sft_gvp_name_conversion_fewshot_ngram3|true|2||"
 "${CHECKPOINT_DIR:-/path/to/checkpoints}/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/name_conversion/checkpoint-268|qwen3_8b_cpt_sft_gvp_name_conversion_nofewshot_ngram3|false|3||"
)

# ==================== function ====================

# model pathgenerateoutput
generate_output_name() {
 local model_path=$1
 local fewshot=$2
 
 # removepathprefixkey
 local name=$(echo "$model_path" | sed 's|.*/checkpoint/||' | sed 's|.*/model/||' | sed 's|/|_|g')
 
 #
 name=$(echo "$name" | sed 's/[^a-zA-Z0-9_-]/_/g')
 
 # fewshotsuffix
 if [ "$fewshot" = "true" ]; then
 name="${name}_fewshot"
 else
 name="${name}_nofewshot"
 fi
 
 echo "$name"
}

# parsetaskconfig
parse_task() {
 local task=$1
 IFS='|' read -r model_path output_name fewshot gpu tasks extra_args <<< "$task"
 
 # ifoutputemptygenerate
 if [ -z "$output_name" ]; then
 output_name=$(generate_output_name "$model_path" "$fewshot")
 fi
 
 # if tasks emptysetemptystringrunalltask
 if [ -z "$tasks" ]; then
 tasks=""
 fi
 
 echo "$model_path|$output_name|$fewshot|$gpu|$tasks|$extra_args"
}

# parseparameter
parse_extra_args() {
 local extra_args=$1
 local args=""
 
 if [ -n "$extra_args" ]; then
 IFS=',' read -ra PARAMS <<< "$extra_args"
 for param in "${PARAMS[@]}"; do
 if [[ "$param" == *"="* ]]; then
 IFS='=' read -r key value <<< "$param"
 args="${args} --${key} ${value}"
 fi
 done
 fi
 
 echo "$args"
}

# runevaluationtask
run_evaluation() {
 local model_path=$1
 local output_name=$2
 local fewshot=$3
 local gpu=$4
 local tasks=$5
 local extra_args=$6
 
 local model_output="${OUTPUT_BASE_DIR}/${output_name}"
 mkdir -p "${model_output}"
 
 # build
 local cmd="CUDA_VISIBLE_DEVICES=${gpu} uv run --preview-features extra-build-dependencies python eval/eval_smolinstruct.py"
 cmd="${cmd} --raw_data_dir \"${RAW_DATA_DIR}\""
 cmd="${cmd} --template_dir \"${TEMPLATE_DIR}\""
 cmd="${cmd} --output_dir \"${model_output}\""
 cmd="${cmd} --molaware_ckpt \"${model_path}\""
 cmd="${cmd} --token_classifier_path \"${TOKEN_CLS_PATH}\""
 cmd="${cmd} --realtime_mol ${DEFAULT_REALTIME_MOL}"
 cmd="${cmd} --max_new_tokens ${DEFAULT_MAX_NEW_TOKENS}"
 cmd="${cmd} --temperature ${DEFAULT_TEMPERATURE}"
 cmd="${cmd} --top_p ${DEFAULT_TOP_P}"
 cmd="${cmd} --repetition_penalty ${DEFAULT_REPETITION_PENALTY}"
 cmd="${cmd} --no_repeat_ngram_size ${DEFAULT_NO_REPEAT_NGRAM_SIZE}"
 cmd="${cmd} --data_limit ${DEFAULT_DATA_LIMIT}"
 
 # fewshotparameter
 if [ "$fewshot" = "true" ]; then
 cmd="${cmd} --few_shot ${DEFAULT_FEW_SHOT}"
 cmd="${cmd} --few_shot_dir \"${DEV_DATA_DIR}\""
 cmd="${cmd} --few_shot_seed ${DEFAULT_FEW_SHOT_SEED}"
 fi
 
 # taskchooseparameter
 if [ -n "$tasks" ]; then
 cmd="${cmd} --include_tasks \"${tasks}\""
 fi
 
 cmd="${cmd} --prompt_style ${DEFAULT_PROMPT_STYLE}"
 cmd="${cmd} --batch_size ${DEFAULT_BATCH_SIZE}"
 cmd="${cmd} --disable_verbose_logging"
 # cmd="${cmd} --verbose_gnn"
 cmd="${cmd} --save_json \"${model_output}/metrics.json\""
 cmd="${cmd} --use_flash_attention"
 
 # parameterdefaultvalue
 local parsed_extra=$(parse_extra_args "$extra_args")
 if [ -n "$parsed_extra" ]; then
 cmd="${cmd} ${parsed_extra}"
 fi
 
 # executerecordlog
 echo "[GPU ${gpu}] ============================================================"
 echo "[GPU ${gpu}] evaluationmodel: ${model_path}"
 echo "[GPU ${gpu}] outputdirectory: ${model_output}"
 echo "[GPU ${gpu}] Fewshot: ${fewshot}"
 if [ -n "$tasks" ]; then
 echo "[GPU ${gpu}] evaluationtask: ${tasks}"
 else
 echo "[GPU ${gpu}] evaluationtask: alltask"
 fi
 echo "[GPU ${gpu}] ============================================================"
 
 # setUTF-8encodeenvironmentvariablelogfilecorrectsave
 export PYTHONIOENCODING=utf-8
 export LC_ALL=C.UTF-8
 export LANG=C.UTF-8
 
 # useteeUTF-8encodeoutputwritefile
 eval "${cmd}" 2>&1 | tee -a "${model_output}/evaluation.log"
 
 # ifteefail
 # eval "${cmd}" 2>&1 | python3 -c "import sys; [sys.stdout.buffer.write(line.encode('utf-8', errors='replace') + b'\n') for line in sys.stdin]" | tee "${model_output}/evaluation.log"
 
 local exit_code=${PIPESTATUS[0]}
 if [ $exit_code -eq 0 ]; then
 echo "[GPU ${gpu}] ✅ ${output_name} evaluationcomplete"
 else
 echo "[GPU ${gpu}] ❌ ${output_name} evaluationfail (: $exit_code)"
 fi
 
 return $exit_code
}

# ==================== ====================

# createoutputdirectory
mkdir -p "${OUTPUT_BASE_DIR}"

# checktasklistwhetherempty
if [ ${#TASK_LIST[@]} -eq 0 ]; then
 echo "⚠️ warning: tasklistemptyconfig TASK_LIST"
 echo ""
 echo "configexample"
 echo "declare -a TASK_LIST=("
 echo " \"/path/to/model1||true|6|\""
 echo " \"/path/to/model1||false|7|\""
 echo " \"/path/to/model2|custom_name|true|6|batch_size=8,data_limit=50\""
 echo ")"
 exit 1
fi

# parsealltaskbuildtaskqueue
declare -a PARSED_TASKS=()
for task in "${TASK_LIST[@]}"; do
 parsed=$(parse_task "$task")
 PARSED_TASKS+=("$parsed")
done

# extractalluseGPU
declare -A GPU_SET
for task in "${PARSED_TASKS[@]}"; do
 IFS='|' read -r model_path output_name fewshot gpu tasks extra_args <<< "$task"
 # processGPU
 IFS=',' read -ra GPUS <<< "$gpu"
 for g in "${GPUS[@]}"; do
 GPU_SET[$g]=1
 done
done

# getGPUlist
GPU_LIST=($(printf '%s\n' "${!GPU_SET[@]}" | sort -n))

if [ ${#GPU_LIST[@]} -eq 0 ]; then
 echo "❌ error: validGPUconfig"
 exit 1
fi

echo "============================================================"
echo "🚀 evaluation"
echo "============================================================"
echo "totaltask: ${#PARSED_TASKS[@]}"
echo "useGPU: ${GPU_LIST[*]}"
echo "outputdirectory: ${OUTPUT_BASE_DIR}"
echo ""

# eachGPUprocessPID
declare -A GPU_PIDS
declare -A GPU_TASK_NAMES

# initializeGPUstatus
for gpu in "${GPU_LIST[@]}"; do
 GPU_PIDS[$gpu]=""
 GPU_TASK_NAMES[$gpu]=""
done

FAILED=0
TASK_INDEX=0
TOTAL_TASKS=${#PARSED_TASKS[@]}
step=0 # forparallelstatus

# functionstartnextGPUtask
start_next_task_for_gpu() {
 local gpu=$1
 local start_idx=$2
 
 for ((i=start_idx; i<TOTAL_TASKS; i++)); do
 local task="${PARSED_TASKS[$i]}"
 IFS='|' read -r model_path output_name fewshot task_gpu tasks extra_args <<< "$task"
 
 # checkGPUwhethermatchsupportsGPU
 IFS=',' read -ra TASK_GPUS <<< "$task_gpu"
 for tgpu in "${TASK_GPUS[@]}"; do
 if [ "$tgpu" == "$gpu" ]; then
 echo "[SCHEDULER] GPU ${gpu} starttask: ${output_name}"
 
 # runtask
 run_evaluation "$model_path" "$output_name" "$fewshot" "$gpu" "$tasks" "$extra_args" &
 
 GPU_PIDS[$gpu]=$!
 GPU_TASK_NAMES[$gpu]="$output_name"
 return $i # returnstaskindex
 fi
 done
 done
 return 255 # task
}

# starttaskallGPU
CURRENT_INDEX=0
for gpu in "${GPU_LIST[@]}"; do
 start_next_task_for_gpu $gpu $CURRENT_INDEX
 idx=$?
 if [ $idx -ge 0 ] && [ $idx -lt 255 ]; then
 CURRENT_INDEX=$((idx + 1))
 sleep 2 # startsource
 fi
done
TASK_INDEX=$CURRENT_INDEX

# loop
while [ $TASK_INDEX -lt $TOTAL_TASKS ] || [ -n "$(printf '%s\n' "${GPU_PIDS[@]}" | grep -v '^$')" ]; do
 # allPID
 ACTIVE_PIDS=()
 for gpu in "${GPU_LIST[@]}"; do
 if [ -n "${GPU_PIDS[$gpu]}" ]; then
 ACTIVE_PIDS+=("${GPU_PIDS[$gpu]}")
 fi
 done
 
 if [ ${#ACTIVE_PIDS[@]} -gt 0 ]; then
 # currentparallelruntask
 if [ $((step % 10)) -eq 0 ]; then
 echo "[SCHEDULER] currentparallelrun: ${#ACTIVE_PIDS[@]} task (GPU: $(printf '%s ' "${!GPU_PIDS[@]}"))"
 fi
 step=$((step + 1))
 
 # waittaskcomplete
 wait -n "${ACTIVE_PIDS[@]}" 2>/dev/null
 EXIT_CODE=$?
 
 # GPUtaskcomplete
 for gpu in "${GPU_LIST[@]}"; do
 if [ -n "${GPU_PIDS[$gpu]}" ]; then
 # checkprocesswhetheralreadyend
 if ! kill -0 "${GPU_PIDS[$gpu]}" 2>/dev/null; then
 COMPLETED_GPU=$gpu
 COMPLETED_PID="${GPU_PIDS[$gpu]}"
 COMPLETED_TASK="${GPU_TASK_NAMES[$gpu]}"
 
 # waitprocessendget
 wait "$COMPLETED_PID" 2>/dev/null
 EXIT_CODE=$?
 
 if [ $EXIT_CODE -eq 0 ]; then
 echo "[SCHEDULER] ✅ GPU ${COMPLETED_GPU} taskcomplete: ${COMPLETED_TASK}"
 else
 echo "[SCHEDULER] ❌ GPU ${COMPLETED_GPU} taskfail: ${COMPLETED_TASK} (: $EXIT_CODE)"
 FAILED=$((FAILED + 1))
 fi
 
 # emptyGPUstatus
 GPU_PIDS[$COMPLETED_GPU]=""
 GPU_TASK_NAMES[$COMPLETED_GPU]=""
 
 # ifruntaskstarttaskGPU
 if [ $TASK_INDEX -lt $TOTAL_TASKS ]; then
 start_next_task_for_gpu $COMPLETED_GPU $TASK_INDEX
 new_index=$?
 if [ $new_index -ge 0 ] && [ $new_index -lt 255 ]; then
 TASK_INDEX=$((new_index + 1))
 sleep 1
 else
 # currentGPUtasktaskcontinueloop
 TASK_INDEX=$((TASK_INDEX + 1))
 fi
 fi
 break
 fi
 fi
 done
 else
 # ifruntaskruntaskstartnext
 if [ $TASK_INDEX -lt $TOTAL_TASKS ]; then
 local task="${PARSED_TASKS[$TASK_INDEX]}"
 IFS='|' read -r model_path output_name fewshot task_gpu tasks extra_args <<< "$task"
 
 # choosefirstGPUiftaskGPU
 IFS=',' read -ra TASK_GPUS <<< "$task_gpu"
 local selected_gpu="${TASK_GPUS[0]}"
 
 echo "[SCHEDULER] GPU ${selected_gpu} starttask: ${output_name}"
 run_evaluation "$model_path" "$output_name" "$fewshot" "$selected_gpu" "$tasks" "$extra_args" &
 GPU_PIDS[$selected_gpu]=$!
 GPU_TASK_NAMES[$selected_gpu]="$output_name"
 TASK_INDEX=$((TASK_INDEX + 1))
 sleep 1
 fi
 fi
 
 # CPU
 sleep 1
done

# waitalltaskcomplete
for gpu in "${GPU_LIST[@]}"; do
 if [ -n "${GPU_PIDS[$gpu]}" ]; then
 echo "[SCHEDULER] wait GPU ${gpu} lasttaskcomplete: ${GPU_TASK_NAMES[$gpu]}"
 wait "${GPU_PIDS[$gpu]}"
 EXIT_CODE=$?
 if [ $EXIT_CODE -ne 0 ]; then
 echo "[SCHEDULER] ❌ GPU ${gpu} taskfail: ${GPU_TASK_NAMES[$gpu]} (: $EXIT_CODE)"
 FAILED=$((FAILED + 1))
 else
 echo "[SCHEDULER] ✅ GPU ${gpu} taskcomplete: ${GPU_TASK_NAMES[$gpu]}"
 fi
 fi
done

echo ""
echo "============================================================"
echo "✅ alltaskcomplete"
echo "============================================================"

if [ $FAILED -eq 0 ]; then
 echo "✅ allevaluationtasksuccesscomplete"
 echo "outputdirectory: ${OUTPUT_BASE_DIR}"
 exit 0
else
 echo "⚠️ $FAILED taskfailchecklogfile"
 exit 1
fi
