import argparse
import warnings
import subprocess
import json
import os

MMLU_TASKS = 'hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions'

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_size", type=str, default='7b', 
                        choices=('7b', '13b'),
                        help='size of llama2 chat')
    parser.add_argument("--task", type=str, required=True, 
                        choices=("arc_challenge", "hellaswag", "MMLU", "truthfulqa_mc"))
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--output_path", type=str, default='test', help="a path to put result and log in ./results")
    parser.add_argument("--prompts_file", type=str, required=True, help="A json file containing key 'system_prompt' and 'user_prompt'. ")
    parser.add_argument("--results_file", type=str)
    parser.add_argument("--bnb_quantize", type=bool, default=False)
    args = parser.parse_args()

    return args

def get_script(args, prompts):
    
    script = 'run.sh'
    
    if args.task == 'MMLU':
        task = MMLU_TASKS
    else:
        task = args.task

    
    model = f'pretrained=meta-llama/Llama-2-{args.model_size}-chat-hf'
    use_accelerate='use_accelerate=True'
    args_list = [model, use_accelerate]

    # TODO: add quantization config for llama2
    if args.bnb_quantize:
        quant_args = "load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=bfloat16"
        args_list.append(quant_args)
        model_args = ','.join(args_list)
    else:
        model_args = ','.join(args_list)
    
    
    cmd = f"bash {script} {task} {model_args} {args.few_shot} {args.results_path} {args.prompts_file}"
    return cmd


def launch_cmd(args, script):

    print(f"Running \n: {script}")
    p = subprocess.Popen(script, cwd='./lm-evaluation-harness', shell=True)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError('\n\n'.join((
            f"Log output: {os.path.join(args.results_path, 'eval.log')}",
        )))

def main(args):
    
    ## create results folder it not exist in current folder
    cur_path = os.path.abspath("./")
    results_path = os.path.join(cur_path, 'results', args.output_path)
    args.results_path = results_path
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        
    args.prompts_file = os.path.abspath(args.prompts_file)
    with open(args.prompts_file, 'r') as f:
        prompts = json.load(f)
        
    assert (('system_prompt' in prompts) and ('user_prompt' in prompts))
    
    script = get_script(args, prompts)
    launch_cmd(args, script)
    
    print(f"======= Finish evaluation on {args.task} =======")
    
        
    

if __name__ == "__main__":
    args = parse_args()
    main(args)


