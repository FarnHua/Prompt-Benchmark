# Tasks
We have created the Prompt Benchmark for four tasks, following the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). 

* AI2 Reasoning Challenge (25 shot)
* HellaSwag (10 shot)
* TruthfulQA (0 shot)
* MMLU (5 shot)

We measure prompts on [meta/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) in [Prompt Benchmark]().

# Prompts file
A json file that contains keys 'system_prompt' and 'user_prompt'.
```
{
    "system_prompt":<SYSTEM_PROMPT>,
    "user_prompt":<USER_PROMPT>
}
```

# Quick Start
## Install
```
git clone --recursive https://github.com/FarnHua/Prompt-Benchmark.git
cd Prompt-Benchmark/lm-evaluation-harness
pip install -e .
```
## Test your prompts
```
python3 evaluation.py --task <TASK> --model_size <MODEL_SIZE> --few_shot <FEW_SHOT> --output_path <OUTPUT_PATH> --prompts_file <PROMPT_FILE>
```
* task: Current support 4 tasks: arc_challenge, hellaswag, truthfulqa_mc, MMLU 
* output_path: The results and logs will be under the path: Prompt-Benchmark/results/output_path.
* bnb_quantize: If set ```True```, use following config to quantize model, default ```False```. 
    ```
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
    ```
### Script for verifying prompts in our leaderboard
We will verify your submitted prompt using the following script. The ```few_shot``` argument with be **25, 10, 0, 5** for **ARC, Hellaswag, TruthfulQA and MMLU** as in the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).
```
python3 evaluation.py --task <TASK> --model_size 13b --few_shot <FEW_SHOT> --output_path <OUTPUT_PATH> --prompts_file <PROMPT_FILE>
```
### Example: 
Run ARC on ```meta/Llama-2-7b-chat-hf```
```
python3 evaluation.py --task arc_challenge --model_size 7b --few_shot 25 --output_path arc_test --prompts_file prompt.json
```

Run ARC on ```meta/Llama-2-13b-chat-hf``` with quantization 
```
python3 evaluation.py --task arc_challenge --model_size 13b --few_shot 25 --output_path arc_test_quant --prompts_file prompt.json --bnb_quantize True
```
