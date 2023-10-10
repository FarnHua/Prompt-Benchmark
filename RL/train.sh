export TRANSFORMERS_CACHE=/work/u5273929/huggingface_hub
export HF_DATASETS_CACHE=/work/u5273929/huggingface_hub
export HUGGINGFACE_HUB_CACHE=/work/u5273929/huggingface_hub

python main.py \
    --mode finetune \
    --prompt GPT2 \
    --agent ppo_lm \
    --pretrain_data_path ./pretrain_data/ChatGPT.csv \
    --model_name /work/u5273929/bias-ppo/gpt2_finetune/gpt2-m/gpt2-m-ChatGPT/checkpoint-2985 \
    --dataset example \
    --exp_name 0703Alpaca_innlr9e-6_lmlr_0.1_kl_coef0.1_toxi8gpullk \
    --log_interval 5\
    --seed 42 \
    --bz 2 \
    --ep_lr 1.0 \
    --k_epoch 5\
    --discount_r 1.0 \
    --end_step 300 \
    --sample_time 1 \
    --max_pt_len 30 \
    --inner_lr 9e-6 \
    --lm_lr 0.1 \
    --init_step 0 \
    --save_path 0703Alpaca_innlr9e-6_lmlr_0.1_kl_coef0.1_toxi8gpullk \
    --save_interval 20 \
    --wandb disabled \
    --bot hf-causal-experimental \
    --bot_args pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True \
    --tasks arc_challenge \
    --limit 20 \
    --num_fewshot 0 \
    --no_cache

