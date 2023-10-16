import numpy as np
import torch
import random
import argparse
import importlib
import tqdm
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import wandb
import os
from argparse import Namespace
from torch.utils.data import DataLoader
import yaml
import warnings
from os.path import join
import json
from copy import deepcopy

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    
    # replace yaml default argument to argparser arg. 
    parser = ArgumentParser()
    args  = set_arguments(parser)
    # opt = yaml.load(open(f"configs/{args.config}/config.yaml"), Loader=yaml.FullLoader)
    # opt.update(vars(args))
    fix_seed(args)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.tasks == 'MMLU':
        args.tasks = 'hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions'
    

    if (args.mode is None):
        print("which mode do you want to use: finetune / test")
        raise
    
    wandb.init(mode=args.wandb, project='promptbenchmark-RL', name=args.exp_name, entity='chatbot_ntu')
    wandb.config.update(args)

    agent = importlib.import_module('.module', f"agents.{args.agent}").agent
    prompt = importlib.import_module(".module", f"prompts.{args.prompt}").prompt
    dataset =importlib.import_module(".module", f"dataset.{args.dataset}").dataset
    
    Prompt = prompt(args)
    
    Dataset = dataset(args.pretrain_data_path, Prompt.tokenizer)
    dataloader = DataLoader(Dataset, batch_size=args.bz, shuffle=True, num_workers=0)
    Agent = agent(args, Prompt, dataloader)

    pbar = tqdm(range(args.end_step))
    step = 0

    # for inputs_id, mask, ll in pbar:
    for _ in pbar:
        total_loss = 0
        total_grad = 0
        total_score = 0
        total_mse = 0
        total_pg = 0
        total_entropy = 0
        step +=1
         
        task_bar =tqdm(total=args.sample_time*args.k_epoch,position=1,leave=True)
        task_bar.set_description(desc=f"None, epoch: 0, score:{round(0.0, 3)}, loss:{round(0.0, 5)}",refresh=True)
        
        if step <= args.end_step:
            sample_dicts = []
            scores = []
            ## use input to sample data
            
            for s in range(args.sample_time):
                flatten_dict = Agent.sample_forward(Prompt.model_demo, Prompt.state_network_demo, Prompt.demo_device)
                sample_dicts.append(flatten_dict)

            if args.mode != 'test':
                ## k_epoch means how many times the sample data will be used
                for epoch in range(args.k_epoch):
                    sample_acc_loss = 0
                    sample_acc_score = 0
                    random.shuffle(sample_dicts)
                    for flatten_dict in sample_dicts:
                        
                        loss, score, _, _, _ = Agent.train_forward(flatten_dict, Prompt.train_device)
                        
                        score['score'] /= args.sample_time
                        loss /= args.sample_time
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(Prompt.model.parameters(), 1.0)
                        torch.nn.utils.clip_grad_norm_(Prompt.state_network.parameters(), 1.0)

                        sample_acc_loss += loss
                        sample_acc_score += score['score']
                        task_bar.set_description(desc=f"epoch: {epoch}, score:{round(sample_acc_score / args.bz, 4)}, loss:{round(sample_acc_loss.item() / args.bz, 5)}", refresh=True)
                        task_bar.update(1)

                    Prompt.optimizer.step()
                    Prompt.optimizer.zero_grad()

                
                ## after k_epoch, update demo model 
                if args.update_demo:
                    Prompt.model_demo.load_state_dict(Prompt.model.state_dict())
                    Prompt.state_network_demo.load_state_dict(Prompt.state_network.state_dict())
                sample_dicts = []
                Prompt.model_demo.eval()
                Prompt.state_network_demo.eval()
                task_bar.update(1)

                ## eval on input
                with torch.no_grad():
                    Prompt.state_network.eval()
                    Prompt.model.eval()
                    
                        
                    flatten_dict = Agent.sample_forward(Prompt.model, Prompt.state_network, Prompt.train_device)
                    loss, score, mse, pg_loss, entropy = \
                        Agent.train_forward(flatten_dict, Prompt.train_device)
                    total_loss += loss.item()
                    total_mse += mse
                    total_pg += pg_loss
                    total_entropy += entropy
                    total_score += score['score']
                    scores.append(score)
                    
                    Prompt.state_network.train()
                    Prompt.model.train()
                    
                Prompt.model.zero_grad()
                Prompt.state_network.zero_grad()

                
                tqdm.write(f"outerloss in step {step}: {round(total_loss/args.bz, 4)}")
                tqdm.write(f"outerscore in step {step}: {round(total_score/args.bz, 4)}")
                if args.wandb != 'disabled':
                    Agent.log_wandb(scores, total_loss, total_mse, total_pg, total_entropy, step)
            else:
                total_scores = []
                for flatten_dict in sample_dicts:
                    total_scores.append(flatten_dict)
                if args.log_wandb != 'disabled':
                    Agent.log_wandb(total_scores, 0, 0, 0, 0, step)
            
            if args.write_json:
                
                dest = f"results/{args.save_path}/"
                os.makedirs(dest, exist_ok=True)
                write_path = f'results/{args.save_path}/prompt_score.json'
                records = []
                for score in scores:
                    
                    for i in range(args.bz):
                        records.append(
                            {   
                                'step': step,
                                'loss': total_loss,
                                'prompt': score['model_response'][i],
                                'score': score['predict_list'][i]
                            }
                        )

                with open(write_path, 'a') as njs:
                    json.dump(records, njs, indent=4)
                    
            if step % args.save_interval == 0:
                
                if args.mode == 'test':

                    dest = f"results/{args.save_path}/"
                    os.makedirs(dest, exist_ok=True)
                    with open(f'results/{args.save_path}/checkpoint-step-{step}-output.txt', 'w') as f:
                        for flatten_dict in sample_dicts:
                            prompt=flatten_dict['model_response']
                            for i in range(len(prompt)):
                                sample_prompt = prompt[i]
                                f.write(sample_prompt)   
                                print(sample_prompt)
                else:

                    dest = f"results/{args.save_path}"
                    os.makedirs(dest, exist_ok=True)
                    
                    save_args = deepcopy(args)
                    del save_args.device
                    with open(f'{dest}/args.txt', 'w') as f:
                        json.dump(save_args.__dict__, f, indent=2)

                    model_dest = f'results/{args.save_path}/step{step}'
                    os.makedirs(model_dest, exist_ok=True)
                    
                    Prompt.model.save_pretrained(model_dest)
                    Prompt.tokenizer.save_pretrained(model_dest)
                    torch.save(
                        {k: (v.cpu() if v is not None else None)  # save to cpu tensors
                            for k, v in Prompt.state_network.state_dict().items()},
                        join(model_dest,f'checkpoint-value.pkl'))
                        
        

def set_arguments(parser):
    parser.add_argument("--task", type=str, default="none") # for finetune task
    parser.add_argument("--agent", type=str, default="ppo_ipx")
    parser.add_argument("--config", type=str, default="example")
    parser.add_argument("--prompt", type=str, default="GPT2")
    parser.add_argument("--dataset", type=str, default="Daily") # for finetune task
    parser.add_argument("--pretrain_data_path", type=str, default="./data/netflix_train_key.csv", help="path for dataset")
    parser.add_argument("--mode", type=str, default="test", help="The current option is [ 'finetune', 'test' ]")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--save_path", type=str, default="") # save path
    parser.add_argument("--model_ckpt", type=str, default="") 
    parser.add_argument("--model_name", type=str, default="gpt2-m")
    parser.add_argument("--ratio", type=float, default=4)
    parser.add_argument("--log_interval",type=int, default=50)
    parser.add_argument("--save_interval",type=int, default=25)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--bz", help="batch size", default=8, type=int)
    parser.add_argument("--k_epoch", help="1 batch sample data need to update k times", type=int, default=5)
    parser.add_argument("--discount_r", help="discounted reward coefficient", type=float, default=0.97)
    parser.add_argument("--end_step",type=int, default=500)
    parser.add_argument("--sample_time", type=int, default=8)
    parser.add_argument("--inner_lr", type=float, default=1e-5)
    parser.add_argument("--outer_lr", type=float, default=1e-5)
    parser.add_argument('--max_pt_len', help="maximum prompt length", type=int, default=10)
    parser.add_argument("--mse_lr", type=float, default=1)
    parser.add_argument("--ep_lr", type=float, default=0.01)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--coh_r", type=float, default=0.01)
    parser.add_argument("--lm_lr", type=float, default=0.5)
    parser.add_argument("--num_testing", type=int, default=1)
    parser.add_argument('--update_demo', dest='update_demo', action='store_true')
    parser.add_argument('--no-update_demo', dest='update_demo', action='store_false')
    parser.set_defaults(update_demo=True)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=.9)
    parser.add_argument("--wandb", type=str, default='enabled')
    parser.add_argument("--write_json", type=bool, default=True)
    
    ## lm-evaluation
    parser.add_argument("--bot", required=True, default='hf-causal-experimental')
    parser.add_argument("--bot_args", default="")
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--check_integrity", action="store_true")
    

    args = parser.parse_args()

    return args

def fix_seed(args):
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    return

if __name__ == "__main__":
    main()    