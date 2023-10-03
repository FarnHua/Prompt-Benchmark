from datasets import load_dataset
from tqdm import tqdm
import random

MMLU_TASK = ['high_school_european_history', 'business_ethics', 'clinical_knowledge', 'medical_genetics', 'high_school_us_history', 'high_school_physics', 'high_school_world_history', 'virology', 'high_school_microeconomics', 'econometrics', 'college_computer_science', 'high_school_biology', 'abstract_algebra', 'professional_accounting', 'philosophy', 'professional_medicine', 'nutrition', 'global_facts', 'machine_learning', 'security_studies', 'public_relations', 'professional_psychology', 'prehistory', 'anatomy', 'human_sexuality', 'college_medicine', 'high_school_government_and_politics', 'college_chemistry', 'logical_fallacies', 'high_school_geography', 'elementary_mathematics', 'human_aging', 'college_mathematics', 'high_school_psychology', 'formal_logic', 'high_school_statistics', 'international_law', 'high_school_mathematics', 'high_school_computer_science', 'conceptual_physics', 'miscellaneous', 'high_school_chemistry', 'marketing', 'professional_law', 'management', 'college_physics', 'jurisprudence', 'world_religions', 'sociology', 'us_foreign_policy', 'high_school_macroeconomics', 'computer_security', 'moral_scenarios', 'moral_disputes', 'electrical_engineering', 'astronomy', 'college_biology']
MMLU_SELECTED_TASKID = random.sample(range(len(MMLU_TASK)), 5)

class dataset() :
    def __init__(self, prompt_gen_data): 
        self.data_name = "lukaemon/mmlu"
        self.selected_task = [MMLU_TASK[i] for i in MMLU_SELECTED_TASKID]
        self.prompt_gen_size = prompt_gen_data

    def get_data(self):
        '''
        each data format : [Q, A, label] for each data
        return type : prompt_gen_data, eval_data
        '''
        ret = []
        print("[INFO] : Creating dataset")

        for n in self.selected_task : 
            data = load_dataset("lukaemon/mmlu", n, split='validation')
        
            for x in tqdm(data) : 
                print(x)
                tmp = []

                for _ in ['A', 'B', 'C', 'D'] :
                    if _ == x['target'] : 
                        tmp.append([x['input'], x[_], 1]) 
                    else :
                        tmp.append([x['input'], x[_], 0])
                
                ret.append(tmp)
            
      
        indices = random.sample(range(len(ret)), self.prompt_gen_size)
        
        prompt_gen_data = [ret[i] for i in indices]
        eval_data = [ret[i]for i in range(len(ret)) if i not in indices]


        return prompt_gen_data, eval_data, []


