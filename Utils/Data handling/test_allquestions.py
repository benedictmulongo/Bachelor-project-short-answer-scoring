
import math 
import pandas as pd
import csv
import json
# Import the data object
from data import data 

# Open the data object
question = data()
# Get all answers for question 2 
# RETURN : a list of dictionary for answers for questions 1 
#print(question.get_question(1)[0:10])
# print(" **************** --------------- ***************** ")
# 
# # Get all answers for questions 1 where score = 0 
# # Note that there is an argument score_index where 
# # you can choose either the firt score or the second one
# # score_index = 'score_1'| 'score_2'
# # the default is 'score_1' when not specified 
# # RETURN : a list of text string for all answers found
#print(len(question.get_by_score(quest=0,score=3)))
#print(" **************** --------------- ***************** ")
# 
# # Get all answers string in the form of a list
# # for which score = 0 
# # RETURN : a list of text string for all answers found
print(question.get_text_by_score(quest=0,score=3)[0:4])
# print(" **************** --------------- ***************** ")
# 
# # Get all questions by specifying the index id
# # questions follow by begin then end 
# # RETURN : a list of text string for all answers found 
# # from begin = 1 to end = 3 for question 1 (0)
# print(question.get_by_Id(0,1,3))
# # OR
# # print(question.get_by_Id(quest=0,begin=1, end=3)) 
3