from datasets import load_dataset
from tqdm import tqdm
import random

class dataset() :
    def __init__(self, prompt_gen_data): 
        self.data = load_dataset("hellaswag", split='train')
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

            for i in range(len(x['endings'])) : 
                if i == int(x['label']) : 
                    tmp.append([x['ctx'], x['endings'][i], 1])
                else : 
                    tmp.append([x['ctx'], x['endings'][i], 0])
            
            ret.append(tmp)
        
        indices = random.sample(range(len(ret)), self.prompt_gen_size)
        
        prompt_gen_data = [ret[i] for i in indices]
        eval_data = [ret[i]for i in range(len(ret)) if i not in indices]

        return prompt_gen_data, eval_data, []
        

Dataset = dataset(10)
a, b, c = Dataset.get_data()