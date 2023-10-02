#!/bin/sh
#SBATCH --job-name=ape_hellaswag
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=40G
#SBATCH --account=GOV112004
#SBATCH -o ./log/ape_hellaswag
#SBATCH --ntasks-per-node=1
module purge
module load miniconda3
conda activate llm-eval


export TRANSFORMERS_CACHE=/work/u5273929/huggingface_hub
export HF_DATASETS_CACHE=/work/u5273929/huggingface_hub
export HUGGINGFACE_HUB_CACHE=/work/u5273929/huggingface_hub

mkdir -p test

python main.py \
    --model hf-causal-experimental \
    --model_args 'pretrained=meta-llama/Llama-2-7b-chat-hf' \
    --tasks hellaswag  \
    --num_fewshot 0 \
    --batch_size 2 \
    --limit 50 \
    --output_path test/hellaswag.csv \
    --no_cache --use_prompt # &> $OUTPUT_PATH/eval.log


