from torch.utils.data.dataset import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np 

class dataset(Dataset):
    def __init__(self, path, tokenizer, max_length=62):
        with open(path) as f:
            txt_list = pd.read_csv(path)['sentence'].to_list()
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            if len(txt) >= 150 : continue
            encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True,
                                       max_length=max_length + 3, padding="max_length", return_tensors='pt')
            
            self.input_ids.append(encodings_dict['input_ids'])
            self.attn_masks.append(encodings_dict['attention_mask'])
            
            
        print(f"[INFO] : Using {len(self.input_ids)} data.")
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]
