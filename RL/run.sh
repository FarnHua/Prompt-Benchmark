#!/bin/sh
#SBATCH --job-name=RL-MM
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=40G
#SBATCH --account=GOV112004
#SBATCH --partition=gpNCHC_LLM
#SBATCH -o ./logs/MM
#SBATCH --ntasks-per-node=1


module purge
module load miniconda3
conda activate llm-eval

python main.py \
    --mode finetune \
    --prompt GPT2 \
    --agent ppo_lm \
    --pretrain_data_path ./pretrain_data/netflix_train_key.csv \
    --model_name farnhua/gpt2-small-netflix \
    --dataset example \
    --exp_name MM-test-small-netflix \
    --log_interval 5\
    --seed 42 \
    --bz 4 \
    --ep_lr 1.0 \
    --k_epoch 2\
    --discount_r 1.0 \
    --end_step 100 \
    --sample_time 1 \
    --max_pt_len 30 \
    --inner_lr 9e-5 \
    --lm_lr 0.0 \
    --save_path MM-test-small-netflix \
    --save_interval 10 \
    --wandb disabled \
    --bot hf-causal-experimental \
    --bot_args pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True \
    --tasks hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions \
    --limit 20 \
    --num_fewshot 0 \
    --no_cache
