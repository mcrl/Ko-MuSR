#!/bin/bash

# We utilize LM Evaluation Harness for evaluation
# We assume language model server is already running

# You need to change the following variables

SERVER_HOSTNAME=$(hostname) # change this to the server hostname if running on a different machine
PORT=8000 # change this to the port number if needed
LM_EVAL_PATH="/path/to/lm-evaluation-harness" # change this to your local path of lm-evaluation-harness
OUTPUT_DIR="/path/to/output/dir" # change this to your desired output directory
MUSR_TASK_DIR="musr-tasks" # change this to your local directory of musr yaml configuration files.

task=ko_musr_v3_mm_cot_hint # change this to the desired task. see available tasks in musr-tasks directory.
model_path=Qwen/Qwen3-32B # change this to the model you want to evaluate. Make sure the model is available on your local server.

# ===========================================

max_len=32768
max_gen=8192


export PYTHONPATH="$LM_EVAL_PATH:$PYTHONPATH"

gen_kwargs="temperature=0.6,top_p=0.95,top_k=20" # you can change generation parameters if needed. This is an example for QWEN.

function run_seed(){
    tasks=$1
    n_fewshot=$2
    seed=$3
    
    if [ $n_fewshot -gt 0 ]; then
        fewshot_flag="--fewshot_as_multiturn"
    else
        fewshot_flag=""
    fi

    lm_eval --model local-chat-completions \
        --model_args model=${model_path},max_length=$max_len,seed=$seed,max_gen_toks=$max_gen,num_concurrent=1,tokenized_requests=False,base_url=http://$SERVER_HOSTNAME:$PORT/v1/chat/completions \
        --tasks $tasks \
        --output $OUTPUT_DIR \
        --seed $seed \
        -s \
        --batch_size 1 \
        --num_fewshot $n_fewshot \
        --include_path $MUSR_TASK_DIR \
        --apply_chat_template \
        --gen_kwargs $gen_kwargs \
        $fewshot_flag
}


# wait until server is ready
echo "Waiting for server $SERVER_HOSTNAME:$PORT to be ready..."
while ! curl -s http://$SERVER_HOSTNAME:$PORT/v1/chat/completions > /dev/null; do
    printf "."
    sleep 5
done

# run evaluation

run_seed $task 3 1234