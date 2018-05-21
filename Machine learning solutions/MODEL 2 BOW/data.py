
import math 
import pandas as pd
import csv
import json
#'C:/Users/ben/Desktop/BachelorProject/train.tsv'

class data: 
     # Construct a data object
    def __init__(self, path = 'all_questions.json'):
        self.question_index=['Question_1','Question_2','Question_3','Question_4','Question_5','Question_6','Question_7','Question_8','Question_9','Question_10'] 
        self.f = open(path)
        self.filen = json.load(self.f)
        
    def get_question(self, x=1):
        string = self.question_index[x]
        rt = self.filen[string]
        return rt
    
    def get_by_score(self,quest=1,score=3,score_index='score_1'):
        string = self.question_index[quest]
        rt = self.filen[string]
        ret = [x for x in rt if x[score_index] == str(score) ]
        return ret
        
    def get_text_by_score(self,quest=1,score=3,score_index='score_1'):
        string = self.question_index[quest]
        rt = self.filen[string]
        ret = [x['text'] for x in rt if x[score_index] == str(score) ]
        return ret
        
    def get_by_Id(self,quest=1,begin=1, end=27588) :
        string = self.question_index[quest]
        rt = self.filen[string]
        ret = [x for x in rt if (begin <= int(x['id']) <= end) ]
        return ret
        