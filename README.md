# Tasks
We have created the Prompt Benchmark for four tasks, following the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). 

* AI2 Reasoning Challenge (ARC)
* HellaSwag
* TruthfulQA 
* MMLU

We measure prompts on [meta/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) in Prompt Benchmark.

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
* task: [arc_challenge, hellaswag, truthfulqa_mc, MMLU] 
* output_path: The results and logs will be under the path: Prompt-Benchmark/results/output_path.

### Example: 
Run ARC on meta/Llama-2-7b-chat-hf
```
python3 evaluation.py --task arc_challenge --model_size 7b --few_shot 0 --output_path arc_test --prompts_file prompt.json
```
