#!/bin/bash

pip install -r requirements.txt
git config --global credential.helper store
git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git
cd bigcode-evaluation-harness
pip install -e .

# Variables
MODELS=(
        "NinedayWang/PolyCoder-160M"
        "NinedayWang/PolyCoder-2.7B"
        "NinedayWang/PolyCoder-0.4B"
        "bigcode/starcoder2-15b"
        "bigcode/starcoder2-7b"
        "bigcode/starcoder2-3b"
        "codellama/CodeLlama-13b-hf"
        "codellama/CodeLlama-7b-hf"
        "deepseek-ai/deepseek-coder-6.7b-base"
        "deepseek-ai/deepseek-coder-1.3b-base"
        "facebook/incoder-6B"
        "facebook/incoder-1B" 
        "google/codegemma-7b"
        "google/codegemma-2b"
        "Qwen/CodeQwen1.5-7B"
        "WizardLMTeam/WizardCoder-15B-V1.0"
        "Salesforce/codegen25-7b-mono_P"
)

TEMPERATURE="0.8"
MAX_LENGTH="128"
BATCH_SIZE="100"
N_SAMPLES="100"
TASKS=(
  "conala"
  "ds1000-all-completion"
  "humaneval"
  "mbpp"
  )
ROOT_DIR="/workspace"

for MODEL in "${MODELS[@]}"; do
  cd ${ROOT_DIR}/bigcode-evaluation-harness
  FT_SAVE_PATH="${ROOT_DIR}/ft_models/${MODEL}"


  for TASK in "${TASKS[@]}"; do
    cd ${ROOT_DIR}/bigcode-evaluation-harness
    
    echo "Evaluating samples for base model: ${MODEL}, task: ${TASK}"
    if [ "${TASK}" = "ds1000-all-completion" ]
    then
    accelerate launch  main.py\
      --model ${MODEL} \
      --task ${TASK} \
      --n_samples ${N_SAMPLES}\
      --temperature ${TEMPERATURE} \
      --allow_code_execution \
      --load_generations_path "${GENERATION_SAVE_PATH}/samples_${TEMPERATURE}_${TASK}.json"\
      --metric_output_path  "${GENERATION_SAVE_PATH}/results_${TEMPERATURE}.json"
    elif [ "${TASK}" = "humaneval" ] || [ "${TASK}" = "mbpp" ]
    then
      cd ${ROOT_DIR}
      python evaluation.py \
      --task ${TASK} \
      --model ${MODEL} \
      --sample_path  "${GENERATION_SAVE_PATH}/samples_${TEMPERATURE}_${TASK}.json"
    fi

    GENERATION_FT_SAVE_PATH="${ROOT_DIR}/result_ft/${TASK}/${MODEL}"
    mkdir -p ${GENERATION_FT_SAVE_PATH}

    echo "Generating samples for fine-tuned model: ${MODEL}, task: ${TASK}"

    if [ "${TASK}" = "conala" ]
    then
      accelerate launch  main.py \
      --model ${MODEL} \
      --peft_model ${FT_SAVE_PATH} \
      --max_length_generation ${MAX_LENGTH} \
      --tasks ${TASK} \
      --do_sample True \
      --n_samples 1 \
      --temperature ${TEMPERATURE} \
      --batch_size ${BATCH_SIZE}\
      --precision fp16 \
      --do_sample True \
      --use_auth_token \
      --load_in_4bit\
      --save_generations_path "${GENERATION_FT_SAVE_PATH}/samples_${TEMPERATURE}.json"\
      --metric_output_path  "${GENERATION_FT_SAVE_PATH}/results_${TEMPERATURE}.json"
    else
      accelerate launch main.py \
        --model ${MODEL} \
        --peft_model ${FT_SAVE_PATH} \
        --tasks ${TASK} \
        --max_length_generation ${MAX_LENGTH} \
        --temperature ${TEMPERATURE}\
        --do_sample True \
        --n_samples ${N_SAMPLES} \
        --batch_size ${BATCH_SIZE} \
        --precision fp16\
        --load_in_4bit\
        --allow_code_execution \
        --generation_only \
        --save_generations \
        --use_auth_token \
        --trust_remote_code\
        --save_generations_path "${GENERATION_FT_SAVE_PATH}/samples_${TEMPERATURE}.json"
      fi

      echo "Evaluating samples for fine-tuned model: ${MODEL}, task: ${TASK}"
      if [ "${TASK}" = "ds1000-all-completion" ]
      then
      accelerate launch  main.py\
        --model ${MODEL} \
        --tasks ${TASK} \
        --n_samples ${N_SAMPLES}\
        --temperature ${TEMPERATURE} \
        --allow_code_execution \
        --load_generations_path "${GENERATION_FT_SAVE_PATH}/samples_${TEMPERATURE}_${TASK}.json"\
        --metric_output_path  "${GENERATION_FT_SAVE_PATH}/results_${TEMPERATURE}.json"
      elif [ "${TASK}" = "humaneval" ] || [ "${TASK}" = "mbpp" ]
      then
        cd ${ROOT_DIR}
        python evaluation.py \
        --task ${TASK} \
        --model ${MODEL} \
        --sample_path  "${GENERATION_FT_SAVE_PATH}/samples_${TEMPERATURE}_${TASK}.json"
      fi
  done
done

