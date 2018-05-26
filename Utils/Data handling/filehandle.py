
import pandas as pd
import csv
import json
from spell_check_API import *
#'C:/Users/ben/Desktop/BachelorProject/train.tsv'

#Initialization of the dictionary
all_questions = {"Question_1":[],"Question_2":[],"Question_3":[],"Question_4":[],"Question_5":[],"Question_6":[],"Question_7":[],"Question_8":[],"Question_9":[],"Question_10":[]}

#Simple dict to update the value while iterating 
idem = {"id":1,"set":1,"score_1":3,"score_2":"2","text":" "}

#Function to read the file and update the idem and all_questions
def read_tsv(index = 1, keys = 'Question_1'):
    print(" *************** Begin ",index," ********************** ")
    with open('C:/Users/ben/Desktop/BachelorProject/train.tsv') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            idem = {"id":1,"set":1,"score_1":3,"score_2":"2","text":" "}
            if(row[1] == str(index)):
                #Update idem
                idem['id'] = row[0]
                idem['set'] = row[1]
                idem['score_1'] = row[2]
                idem['score_2'] = row[3]
                idem['text'] = correct(row[4])
                
                #Store idem in the all_questions dictionary
                all_questions[keys].append(idem)
    
    print(" *************** Finnished ",index ," ********************** ")

#Call the function to save information in dict
Q=['Question_1','Question_2','Question_3','Question_4','Question_5','Question_6','Question_7','Question_8','Question_9','Question_10']
count = 0
for x in Q:
    print(x)
    count = count + 1
    read_tsv(index = count, keys = x)

#Save to file
f = open('all_questions.json', 'w')
json.dump(all_questions, f, indent=2)
f.close()
