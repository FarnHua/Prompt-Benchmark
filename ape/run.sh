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


