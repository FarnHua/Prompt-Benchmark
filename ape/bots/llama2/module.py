import os
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from torch import nn
import numpy as np

class bot(nn.Module):
    def __init__(self, config):
        super().__init__()

        model_name = 'meta-llama/Llama-2-7b-chat-hf'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                    device_map="auto", 
                                                    torch_dtype=torch.float16)
        model.config.pad_token_id = self.tokenizer.eos_token_id
        self.lm = model
        self.lm.eval()
        
        self.generation_args = dict(temperature=0.0, max_new_tokens=256, repetition_penalty=1.1, output_scores=True, return_dict_in_generate=True)

    def get_prompt(self, sentence, system_prompt=None):
        ## modified from https://huggingface.co/spaces/huggingface-projects/llama-2-13b-chat/blob/main/model.py#L24
        if not system_prompt:
            system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
        
        texts.append(f"{sentence} [/INST]")

        return ''.join(texts)
    

    def make_response(self, prefix_sentences):
        with torch.no_grad():
            sentences = []
            seg = ''
            for i in range(len(prefix_sentences)):
                # prompt = self.get_prompt(prefix_sentences[i])
                prompt = prefix_sentences[i]
                sentences.append(prompt)
            
            inputs = self.tokenizer(sentences, return_tensors="pt", padding=True).to(self.lm.device)    
            outputs = self.lm.generate(**inputs, **self.generation_args)
            reply_strings = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        tmp = []
        for i in range(len(prefix_sentences)) :
            tmp.append(reply_strings[i].split('[/INST]')[-1].strip())
        return tmp
    
