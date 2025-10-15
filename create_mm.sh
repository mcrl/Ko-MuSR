#!/bin/bash

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Please set your OpenAI API key in the OPENAI_API_KEY environment variable."
    exit 1
fi


export PYTHONPATH=$(pwd):$PYTHONPATH

cd musr_dataset_scripts

num_samples=1 # change this as you want

function run_model(){
    model=$1
    seed=$2
    python create_murder_mysteries.py --cache --model $model --validation_tries 5 --num_samples $num_samples --seed $seed 2>&1 | tee "logs/log_mu_${model}_${seed}.txt"
}

seed=1234 # for running multiple scripts in parallel. change this as you want

mkdir -p logs
run_model gpt-4o $seed 
