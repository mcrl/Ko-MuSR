# Ko-MuSR

Codes for the paper **Ko-MuSR: A Multistep Soft Reasoning Benchmark for LLMs Capable of Understanding Korean**

## Tested Environment

```plain
openai==1.78.0
lm-eval==0.4.8 # Installed with [api] configuration
```

## Setup

### 1. Installing required packages

```bash
pip install -r requirements.txt
```

Installing LM Evaluation Harness (Use api configuration)

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e ".[api]"
```

### 2. Other requirements that might be of your interest

1. For data synthesis, please make your openai API key available with the environment variable, `OPENAI_API_KEY`
```bash
export OPENAI_API_KEY="<your-openai-api-key>"
```
2. For evaluation, prepare your language model servers. Offline inference using lm-evaluation-harness might work, but this feature is not tested.


## Reproducing data

### 1. Generating random domain

For generating random domain, utilize two scripts

```bash
# before running the script, make sure you setup the openai api key here!
export OPENAI_API_KEY="<your-openai-api-key"

python sample_madlib_op.py # for generating random domain for object placements task.
python sample_madlib_ta.py # for generating random domain for team allocation task.
```

### 2. Generating data instances

For generating data instances, utilize three scripts

```bash
# before running the script, make sure you setup the openai api key here!
export OPENAI_API_KEY="<your-openai-api-key"

bash create_mm.sh # for generating murder mysteries data
bash create_op.sh # for generating object placements data
bash create_ta.sh # for generating team allocations data
```

## Evaluation

We utilize LM Evaluation Harness for the evaluation. We provide task descriptions in `musr-tasks` directory.

We provide evaluation example in `evaluation.sh` script that utilizes local inference servers. See `lm-evaluation-harness` repository for advanced usage.

## License

This repository is licensed under MIT license. For original code, see [MuSR Repository](https://github.com/Zayne-sprague/MuSR)