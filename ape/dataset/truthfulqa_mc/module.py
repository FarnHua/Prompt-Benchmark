from datasets import load_dataset
from tqdm import tqdm
import random

class dataset() :
    def __init__(self, prompt_gen_data): 
        self.data = load_dataset("truthful_qa", 'multiple_choice', split='validation')
        self.prompt_gen_size = prompt_gen_data

    def get_data(self):
        '''
        each data format : [Q, A, label] for each data
        return type : prompt_gen_data, eval_data
        '''
        ret = []
        print("[INFO] : Creating dataset")
        for x in tqdm(self.data) : 
            tmp = []
            for i in range(len(x['mc2_targets']['choices'])) : 
                tmp.append([x['question'], x['mc2_targets']['choices'][i], x['mc2_targets']['labels'][i]])
            ret.append(tmp)
        
        indices = random.sample(range(len(ret)), self.prompt_gen_size)
        
        prompt_gen_data = [ret[i] for i in indices]
        eval_data = [ret[i]for i in range(len(ret)) if i not in indices]

        return prompt_gen_data, eval_data, indices
        
