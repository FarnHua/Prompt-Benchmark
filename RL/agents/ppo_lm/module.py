import torch
from torch import nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import numpy as np
import re
import wandb
from copy import deepcopy
from transformers import TopPLogitsWarper, TopKLogitsWarper
import json
import sys
sys.path.append('../../lm_eval')
from lm_eval import tasks, evaluator, utils

class agent(nn.Module):

    def __init__(self, config, prompt, ptx_dataloader):
        super().__init__()

        """
        """
        self.args = config

        device = prompt.device
        self.mode = config.mode
        self.prompt = prompt
        self.eps_clip = config.eps_clip
        
        self.train_task = None
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.top_p = TopPLogitsWarper(top_p=self.top_p)
        self.top_k = TopKLogitsWarper(top_k=self.top_k)
        self.pretrain_dataloader = ptx_dataloader
        self.ptx_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.table = wandb.Table(columns=['step', 'task_name', 'prompt', 'score'])

        if config.tasks is None:
            task_names = tasks.ALL_TASKS
        else:
            task_names = utils.pattern_match(config.tasks.split(","), tasks.ALL_TASKS)

        print(f"Selected Tasks: {task_names}")
        
        ## Only MMLU has more than one task in PromptBenchmark
        if len(task_names) > 1:
            self.task_name = 'MMLU'
        else:
            self.task_name = task_names[0]
        
        lm, task_dict = evaluator.simple_evaluate(
                                    model=config.bot,
                                    model_args=config.bot_args,
                                    tasks=task_names,
                                    num_fewshot=config.num_fewshot,
                                    batch_size=config.bz,
                                    max_batch_size=config.bz,
                                    device=config.device,
                                    no_cache=config.no_cache)
        self.lm = lm
        self.task_dict = task_dict
        

    def sample_forward(self, model, state_net, device=torch.device('cuda:0')):
        
        ## only use the first word in the input sentence
        prev_input = torch.LongTensor([[self.prompt.tokenizer.bos_token_id] * self.args.bz]).squeeze(0).unsqueeze(1).to(device)
        mask = torch.LongTensor([[1] * self.args.bz]).squeeze(0).unsqueeze(1).to(device)
        batch_size = self.args.bz
        append = torch.tensor([[1] for i in range(batch_size)]).to(device)
        temp_sen = [[] for i in range(batch_size)]
    

        ## states: prev_input
        ## action: next token 

        old_states = []
        old_logprobs = []
        old_mask = []
        old_actions = []
        temperature = 1.0
        # mask = torch.cat((mask, append), 1)
        position_ids = mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(mask == 0, 1)
        position_ids = position_ids[:, -1].unsqueeze(-1).to(device)
        eos_index = [0]*batch_size
        past = None

        with torch.no_grad():
            for i in range(self.args.max_pt_len):

                prev_input = prev_input.to(device)
                old_mask.append(mask.detach().cpu())
                old_states.append(prev_input.detach().cpu())
                temp_past = past
                
                output = model(prev_input, past_key_values=temp_past, attention_mask=mask, position_ids=position_ids)
                logits, past = output['logits'], output['past_key_values']
                mask = torch.cat((mask, append), 1)
                position_ids = mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(mask == 0, 1)
                position_ids = position_ids[:, -1].unsqueeze(-1).to(device)
                logits = logits.squeeze(0).squeeze(1)
                soft_logits = logits / temperature
                probs = torch.softmax(soft_logits, dim=-1)
                top_p_top_k_probs = torch.softmax(self.top_p(i, self.top_k(i, soft_logits)), dim=-1)
                dist = Categorical(probs)
                dist2 = Categorical(top_p_top_k_probs) 
                prev_input = dist.sample()[:, None]
                old_actions.append(prev_input.detach().cpu())
                old_logprobs.append(dist.log_prob(prev_input.squeeze()).detach().cpu())
                
                for j in range(batch_size):
                    origin_index = j % batch_size
                    temp_sen[j].append(prev_input[origin_index].item())
                    
                
        ##########################################################################################
        eos_index = [len(temp_sen[0]) for j in range(len(temp_sen))]
        
        end_index = self.prompt.tokenizer.eos_token_id
        
        for j in range(len(temp_sen)):
            if end_index in temp_sen[j]:
                eos_index[j] = temp_sen[j].index(end_index)
                temp_sen[j] = temp_sen[j][:eos_index[j]]
                
        model_response = [self.prompt.tokenizer.decode(x, skip_special_tokens=True) for x in temp_sen]
        predict_list = self.get_reward(model_response)
        
        score = 0
        tempscore = []
        step = 0
        for s in predict_list:
            score += s
            tempscore.append(s)
        score_emo = np.array(tempscore)
        
        flatten_states = []
        flatten_rewards = []
        flatten_actions = []
        flatten_logprobs = []
        flatten_mask = []
        flatten_values = []

        flatten_states.extend(old_states)
        flatten_logprobs.extend(old_logprobs)
        flatten_actions.extend(old_actions)
        flatten_mask.extend(old_mask)

        flatten_dict = {'flatten_states': flatten_states,
                        'flatten_logprobs': flatten_logprobs,
                        'flatten_actions': flatten_actions,
                        'flatten_mask': flatten_mask,
                        'flatten_rewards': flatten_rewards,
                        'eos_index': eos_index,
                        'score': score,
                        "predict_list":predict_list,
                        "model_response":model_response,
                        'classify_reward': score_emo
                        }

        return flatten_dict

    def train_forward(self, flatten_dict, device=torch.device('cuda:0')):
        
        flatten_states = flatten_dict['flatten_states']
        flatten_logprobs = flatten_dict['flatten_logprobs']
        flatten_actions = flatten_dict['flatten_actions']
        
        flatten_mask = flatten_dict['flatten_mask']
        
        eos_index = flatten_dict['eos_index']
        
        score_emo = flatten_dict['classify_reward']
        batch_size = self.args.bz
        past = None
        eps_clip = self.eps_clip
        mse = 0
        true_total_mse = 0
        entropy = 0
        pg_loss = 0
        #calculate all reward mean and variance
        outter_count = 0
        loss = 0

        flatten_all = []
        logits_list = []
        prediction_list = []
        
        length_list = [1 for _ in range(len(flatten_states[0]))]
        

        for num in range(len(flatten_states)):
            flatten_states[num] = flatten_states[num].to(device)
            flatten_logprobs[num] = flatten_logprobs[num].to(device)
            flatten_actions[num] = flatten_actions[num].to(device)
            flatten_mask[num] = flatten_mask[num].to(device)
            position_ids = flatten_mask[num].long().cumsum(-1) - 1
            position_ids.masked_fill_(flatten_mask[num] == 0, 1)
            position_ids = position_ids[:, -1].unsqueeze(-1).to(device)
            temp_past = past
            output = self.prompt.model(flatten_states[num], past_key_values=temp_past, attention_mask=flatten_mask[num], position_ids=position_ids)
            
            logits, past = output['logits'], output['past_key_values']
            
            probs = torch.softmax(logits.squeeze(0).squeeze(1), dim=-1)
            
            dist = Categorical(probs)
            
            for j in range(batch_size):
                act = flatten_actions[num][j]
                
            
            hidden_states = self.prompt.model.transformer(flatten_states[num],past_key_values=temp_past, attention_mask=flatten_mask[num], position_ids=position_ids)[0]      
            hidden = self.prompt.state_network(hidden_states)
            prediction_list.append(hidden)
            logits_list.append(logits) 
        
        flatten_rewards, r_mean, r_std = self.prepare_reward(score_emo, eos_index, batch_size)
        outter_count = 0
        for num in range(len(flatten_states)):
            prediction = prediction_list[num]
            actionprobs = F.softmax(logits_list[num], dim=-1)
            rewards_tensor = torch.tensor(flatten_rewards[num]).to(device)
            rewards_norm = (rewards_tensor - r_mean) / (r_std + 1e-9) + r_mean

            dist = Categorical(actionprobs)
            action_logprobs = dist.log_prob(flatten_actions[num])
            dist_entropy = dist.entropy()

            ratios = torch.exp(action_logprobs.squeeze() - flatten_logprobs[num])
            advantages = rewards_norm - prediction.squeeze().detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            mseloss=nn.MSELoss()
            index_loss = 0
            index_mse = 0
            index_pg = 0
            cur_size = 0
            for i in range(batch_size):
                if num < eos_index[i]:
                    index_mse += torch.mean(self.args.mse_lr * mseloss(
                    prediction.squeeze()[i].float(), rewards_norm[i].float()))
                    index_pg += torch.mean(-torch.min(surr1[i].float(), surr2[i].float()))
                    cur_size += 1
                    outter_count += 1
                    length_list[i] += 1

            if cur_size == 0:
                break
            mse += index_mse
            entropy += torch.mean(-dist_entropy).item()
            pg_loss += index_pg 
        pg_loss /= (outter_count + 1e-9)
        mse /= (outter_count + 1e-9)
        loss += pg_loss + mse# - self.args.ep_lr * entropy
        

        if self.args.lm_lr != 0: 
            inputs_id, mask = next(iter(self.pretrain_dataloader))
            inputs_id = inputs_id.to(device)
            mask = mask.to(device)
            labels_id = deepcopy(inputs_id)
            labels_id.masked_fill_(mask == 0, -100)
            outputs = self.prompt.model(inputs_id, attention_mask=mask, labels=labels_id)
            lm_loss = outputs['loss']
            loss = lm_loss * self.args.lm_lr  + (1- self.args.lm_lr) * loss
            
            
        flatten_dict["length"] = length_list
    
       
        if self.args.lm_lr != 0:
            flatten_dict['lm_loss'] = lm_loss.item()
        else:
            flatten_dict['lm_loss'] = 0

        return loss, flatten_dict, mse.item(), pg_loss.item(), entropy

    def prepare_reward(self, score_emo, eos_index, batch_size):
        batchwise_pt_len_rewards= []
        reward_collect = []
        r_mean = 0
        r_std = 0

        rewards = [[] for i in range(batch_size)]

        for i in range(batch_size):
            reward = score_emo[i]
            num = self.args.max_pt_len if eos_index[i] >= self.args.max_pt_len else eos_index[i] 
            discount_reward = 0
            for j in range(num):
                if j == 0:
                    discount_reward = reward + self.args.discount_r * discount_reward
                else:
                    discount_reward = self.args.discount_r * discount_reward
                rewards[i].append(discount_reward)
                reward_collect.append(discount_reward)
            rewards[i].reverse()
            while len(rewards[i]) < self.args.max_pt_len:
                rewards[i].append(0)
        
        reward_collect = np.array(reward_collect)
        r_mean, r_std = np.mean(reward_collect), np.std(reward_collect)

        for i in range(self.args.max_pt_len):
            batch_r = []
            for k in range(batch_size):
                batch_r.append(rewards[k][i])   
            batchwise_pt_len_rewards.append(batch_r[:])
        flatten_rewards = []
        flatten_rewards.extend(batchwise_pt_len_rewards)
        return flatten_rewards, r_mean, r_std
        
    

    def get_reward(self, prompts):
        
        with torch.no_grad():
            scores = []
            for prompt in prompts:
                results = evaluator.evaluate(
                    lm=self.lm,
                    task_dict=self.task_dict,
                    num_fewshot=self.args.num_fewshot,
                    limit=self.args.limit,
                    write_out=False,
                    system_prompt='',
                    user_prompt=prompt
                )['results']
                
                if self.task_name in ['truthfulqa_mc'] :
                    scores.append(results['truthfulqa_mc']['mc2'])
                    
                elif self.task_name in ['hellaswag'] :
                    scores.append(results['hellaswag']['acc_norm'])
                    
                elif self.task_name in ['MMLU'] :
                    tmp = [results[key]['acc'] for key in results]
                    scores.append(np.mean(tmp))
                    
                elif self.task_name in ['arc_challenge'] :
                    scores.append(results['arc_challenge']['acc_norm'])
        return scores


    def log_wandb(self, flatten_dicts, total_loss, total_mse, total_pg, total_entropy, batch):
        meta_total = len(flatten_dicts)
        training_score = 0
        coherence_score = 0
        control_score = 0
        lm_loss = 0

        for score in flatten_dicts:
            training_score += score['score']
            lm_loss += score['lm_loss']

        wandb.log({'outerloss': total_loss / meta_total , \
                    'outermse': total_mse / meta_total, \
                    'outerpg': total_pg / meta_total, \
                    'outerentropy': total_entropy / meta_total, \
                    'outerscore': training_score / self.args.bz / meta_total, \
                    'lm_loss': lm_loss / meta_total}, \
                    step=batch)
        if batch % 2 == 0:
            for flatten_dict in flatten_dicts:
                prompt=flatten_dict['model_response']
                predict_list=flatten_dict['predict_list']
                for i in range(len(prompt)):
                    sample_prompt = prompt[i]
                    score = predict_list[i]
                    self.table.add_data(batch, self.task_name, sample_prompt, score)
                    
            
            new_table = wandb.Table(
                columns=self.table.columns, data=self.table.data
            )
            wandb.log({"prompt_score": new_table})
            
