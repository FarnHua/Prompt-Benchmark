#!/bin/sh
#SBATCH --job-name=check_MMLU_8gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=40G
#SBATCH --account=GOV112004
#SBATCH -o ./log/check_MMLU_8gpu
#SBATCH --ntasks-per-node=1

module purge
module load miniconda3
conda activate llm-eval

# hellaswag
# python3 evaluation.py --task hellaswag --model_size 13b --few_shot 10 --output_path hella_APE_check --prompts_file hand_crafted_prompts/ape_hella.json

# MMLU
python3 evaluation.py --task MMLU --model_size 13b --few_shot 5 --output_path MMLU_hand_13b --prompts_file hand_crafted_prompts/hand_MMLU.json

# truthfulqa
# python3 evaluation.py --task truthfulqa_mc --model_size 13b --few_shot 0 --output_path truth_ape --prompts_file hand_crafted_prompts/ape_truth.json