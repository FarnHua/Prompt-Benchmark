import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from torch.optim import AdamW
from copy import deepcopy
from os.path import exists

class prompt():
    def __init__(self, config):
        self.args = config
        self.device = self.train_device = self.demo_device = config.device
        self.configuration = GPT2Config.from_pretrained(config.model_name)
        hidden_size = self.configuration.n_embd
        self.state_network = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_size, 1))
        self.state_network_demo = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_size, 1))


        if config.model_ckpt != '':

            print(f"[Load LM from saved point]: the original path: results/{config.model_ckpt}/checkpoint-step-{self.args.init_step}-prompt.pkl")
            print(f"[Load VM from saved point]: the original path: results/{config.model_ckpt}/checkpoint-step-{self.args.init_step}-value.pkl")
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium', bos_token='<|startoftext|>',eos_token='<|endoftext|>', pad_token='<|pad|>')
            self.model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.load_state_dict(torch.load("./results/" + config.model_ckpt + f"/checkpoint-step-{self.args.init_step}-prompt.pkl"))
    

            self.model_demo = GPT2LMHeadModel.from_pretrained("gpt2-medium")
            self.model_demo.resize_token_embeddings(len(self.tokenizer))
            self.model_demo.load_state_dict(torch.load("./results/" + config.model_ckpt + f"/checkpoint-step-{self.args.init_step}-prompt.pkl"))

            self.state_network.load_state_dict(torch.load("results/" + config.model_ckpt + f"/checkpoint-step-{self.args.init_step}-value.pkl"))
            self.state_network_demo.load_state_dict(torch.load("results/" + config.model_ckpt + f"/checkpoint-step-{self.args.init_step}-value.pkl"))

        else:
            self.configuration = GPT2Config.from_pretrained('gpt2-medium')
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')  
            self.model = GPT2LMHeadModel.from_pretrained(config.model_name)
            self.model_demo = GPT2LMHeadModel.from_pretrained(config.model_name)
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model_demo.resize_token_embeddings(len(self.tokenizer))
            
        self.optim_param = list(self.model.named_parameters())
        no_decay = ['bias', 'ln']   # no decay for bias and LayerNorm (ln)
        self.optimizer_grouped_parameters = [
        {'params': [p for n, p in self.optim_param
                    if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01},
        {'params': [p for n, p in self.optim_param
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        
        self.optimizer =  AdamW(self.optimizer_grouped_parameters, self.args.inner_lr)

        self.model.to(self.device)
        self.model_demo.to(self.device)
        self.state_network.to(self.device)
        self.state_network_demo.to(self.device)
        self.model.train()
        self.state_network.train()
        
        self.model_demo.eval()
        self.state_network_demo.eval()
    
