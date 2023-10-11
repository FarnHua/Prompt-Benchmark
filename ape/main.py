import numpy as np
import torch
import random
import argparse
import importlib
import tqdm
from tqdm import tqdm
from argparse import ArgumentParser
from argparse import Namespace
import yaml
import warnings
from os.path import join
import json
import pandas as pd
from lm_eval import tasks, evaluator, utils

warnings.simplefilter(action='ignore', category=FutureWarning)

MMLU_TASKS = 'hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions'


def create_llm_input(data, demo_template, prompt_template, subsample_num) :
    '''
    input : sentence, templates. 
    return : intructions to construct prompts. 
    '''
    indices = random.sample(range(len(data)), subsample_num)
    subsampled_data = [data[i] for i in indices]

    demo = ""
    for i in range(len(subsampled_data)) :
        for j in range(len(subsampled_data[i])) :
            tmp = subsampled_data[i][j]
            if tmp[-1] == 1 :
                demo = demo + demo_template.replace('[INPUT]', tmp[0]).replace('[OUTPUT]', tmp[1])
                break
                
    full_template = prompt_template.replace('[full_DEMO]', demo)

    return full_template


def main():
    
    parser = ArgumentParser()
    args  = set_arguments(parser)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset =importlib.import_module(".module",f"dataset.{args.tasks}").dataset
    Dataset = dataset(args.prompt_gen_size)
    if args.tasks == 'MMLU' :
        task = MMLU_TASKS
    else :
        task = args.tasks
    ### get dataset and bot
    bot = importlib.import_module(".module",f"bots.llama2").bot
    Bot = bot({})

    ### get two splits of data
    prompt_gen_data, eval_data, prompt_gen_indices = Dataset.get_data()
    print(f"[INFO] : {len(prompt_gen_data)} data for creating prompts, {len(eval_data)} for evaluating.")

    ### get prompts
    eval_template = "Instructions: [PROMPT].\n\nQ: [INPUT]\nA: [OUTPUT]"
    prompt_gen_template = "You are given the following instructions: Please provide an instruction for answering the following questions more effectively.\n\n[full_DEMO]"
    demos_template = "Q: [INPUT]\nA: [OUTPUT]"

    tmp_prompts = []
    print("[INFO] : Generating prompts from llama2 ...")
    for i in tqdm(range(args.num_prompts)) : 
        tmp = create_llm_input(prompt_gen_data, demos_template, prompt_gen_template, args.subsample_num)
        tmp_prompts.append(tmp)
    

    prompts = Bot.make_response(tmp_prompts)

    ape_result = []
   
    for i in range(args.num_iter) :
    ##### start evaluate
        tmp_result = []
        for j in range(len(prompts)) :
            tmp_dict = {}
            tmp_dict['prompt'] = prompts[j]

            if args.tasks is None:
                task_names = tasks.ALL_TASKS
            else:
                task_names = utils.pattern_match(task.split(","), tasks.ALL_TASKS)

            system_prompt = ''
            user_prompt = prompts[j]
            
            description_dict = {}
            results = evaluator.simple_evaluate(
                model=args.model,
                model_args=args.model_args,
                tasks=task_names,
                num_fewshot=args.num_fewshot,
                batch_size=args.batch_size,
                max_batch_size=args.max_batch_size,
                device=args.device,
                no_cache=args.no_cache,
                limit=args.limit,
                description_dict=description_dict,
                decontamination_ngrams_path=args.decontamination_ngrams_path,
                check_integrity=args.check_integrity,
                write_out=args.write_out,
                output_base_path=args.output_base_path,
                use_prompt=args.use_prompt,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                prompt_gen_indices=prompt_gen_indices
            )

            if args.tasks in ['truthfulqa_mc'] :
                score = results['results']['truthfulqa_mc']['mc2']
                tmp_dict['reward'] = score
            elif args.tasks in ['hellaswag'] :
                score = results['results']['hellaswag']['acc_norm']
                tmp_dict['reward'] = score

            elif args.tasks in ['MMLU'] :
                accs = []
                for key in results['results']:
                    accs.append(results['results'][key]['acc'])
                score = np.mean(accs)
                tmp_dict['reward'] = score

            elif args.tasks in ['arc-challenge'] :
                score = results['results']['arc-challenge']['acc_norm']
                tmp_dict['reward'] = score
            tmp_result.append(tmp_dict)

        ape_result.append(tmp_result)
    
    to_write = []
    for x in ape_result[0] : 
        to_write.append([x['reward'], x['prompt']])
        
    df = pd.DataFrame(to_write, columns=['reward', 'prompt'])
    df.to_csv(args.output_path)

    

    
        

    

def set_arguments(parser):
    parser.add_argument("--num_demos", type=int, default=5) #demos to generate prompts
    parser.add_argument("--prompt_gen_size", type=int, default=20) #sampled prompt_gen_size datas from training data
    # parser.add_argument("--task", type=str, default='truthfulqa')
    parser.add_argument("--model", type=str, default='hf-causal-experimental')
    parser.add_argument("--model_args", type=str, default="meta/Llama-2-7b-chat-hf")
    parser.add_argument("--subsample_num", type=int, default=5)
    parser.add_argument("--num_prompts", type=int, default=20)
    parser.add_argument("--num_iter", type=int, default=1)
    parser.add_argument("--system", type=bool, default=False)
    
    # parser.add_argument("--model", required=True)
    # parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None)#, choices=utils.MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--max_batch_size", type=int, default=None,
                        help="Maximal batch size to try with --batch_size auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
    parser.add_argument("--use_prompt", action="store_true", default=False)
    # parser.add_argument("--prompts_file", type=str, required=True)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()    