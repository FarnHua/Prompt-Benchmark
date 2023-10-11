from datasets import load_dataset
from tqdm import tqdm
import random
import pandas as pd

class dataset() :
    def __init__(self, prompt_gen_data): 
        
        self.prompt_gen_size = prompt_gen_data

    def get_data(self):
        '''
        each data format : [Q, A, label] for each data
        return type : prompt_gen_data, eval_data
        '''
        ret = []
        print("[INFO] : Creating dataset")

        Question = pd.read_csv('/work/u5273929/temp/Prompt-Benchmark/ape/dataset/MMLU/mmlu_data.csv')['Question'].to_list()            
        Answer = pd.read_csv('/work/u5273929/temp/Prompt-Benchmark/ape/dataset/MMLU/mmlu_data.csv')['Answer'].to_list()
        
        for i in range(len(Question)) :

            ret.append([[Question[i], Answer[i], 1]])

        indices = random.sample(range(len(ret)), self.prompt_gen_size)
        
        prompt_gen_data = [ret[i] for i in indices]
        eval_data = [ret[i]for i in range(len(ret)) if i not in indices]


        return prompt_gen_data, eval_data, []



