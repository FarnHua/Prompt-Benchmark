# Tasks
Follow Open LLM Leaderboard https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard

* AI2 Reasoning Challenge (ARC)
* HellaSwag
* TruthfulQA 
* MMLU

# Prompt
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
cd lm-evaluation-harness
pip install -e .
```
## Test your prompts
```
python3 evaluation.py --task <TASK> --output_path <OUTPUT_PATH> --prompts_file <PROMPT_FILE>
```
* task: [arc_challenge, hellaswag, truthfulqa_mc, MMLU] 
* output_path: The results and logs will be under the path: Prompt-Benchmark/results/output_path.

### Example: 
```
python3 evaluation.py --task arc_challenge --output_path arc_test --prompts_file prompt.json
```
