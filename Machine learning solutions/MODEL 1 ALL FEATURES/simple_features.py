from nltk import *
import numpy as np
from text_processing import *
from nltk.corpus import wordnet as wn
import json 
from data import data 

question = data()

   
def partial_words_overlap(answer, thresh = 0.7):
    import ngram
    # inspiration -> semsim : a multi-feature approach to semantic 
    # text similarity Adebayo Guidi Boella
    # avg 24
    facit = "know amount vinegar container type sort cup material long time minute temperature experiment sample rinse distilled water size surface area drying method"
    len_answer = len(tokenizer(answer))
    answer = remove_shit(answer)
    answer = remove_stops(tokenizer(answer))
    
    answer = tokenizer(answer)
    #print("Clean answer : ", answer)
    facit = tokenizer(facit)
    
    len_answer_tokenized = len(answer)
    len_facit = len(facit)
    
    G = ngram.NGram(answer) 
    candidate = 0
    cand = []
    sum = 0
    for word in facit:
        candidate = G.search(word, threshold=thresh)
        if len(candidate) > 0:
            cand.append(candidate)
            sum = sum + 1
        
    #return cand, sum, sum /(24 + len_facit),  sum /(len_answe_tokenized  + len_facit)
    return sum /(24 + len_facit), sum /(len_answer_tokenized  + len_facit)
def sem_matrix(answer):
    # Very slow !!!!!!!!!!
    from semantic_similarity import semantic
    sem = semantic()
    
    facit = "know amount vinegar container type cup material time temperature experiment sample rinse distilled water size surface area drying method"
    
    answer = remove_shit(answer)
    answer = remove_stops(tokenizer(answer))
    
    answer = tokenizer(answer)
    #print("Clean answer : ", answer)
    facit = tokenizer(facit)
    
    matrix = [[0] * len(facit) for _ in range(len(answer))]
    for i,x in enumerate(answer):
        for j,y in enumerate(facit):
            alpha = sem.word_sim_rel(x,y)
            #print("word1 = ",x," word2= ",y, " score = ", alpha)
            matrix[i][j] = alpha
    print("Amen !")
    total_x = 0
    total_y = 0
    for i in range(len(answer)):
        max_i = 0
        for j in range(len(facit)):
            if (matrix[i][j] > max_i):
                max_i = matrix[i][j]
                total_x += max_i
 
    for i in range(len(facit)):
        max_i = 0
        for j in range(len(answer)):
            if (matrix[j][i] > max_i):
                max_i = matrix[j][i]
                total_y += max_i
    final = (total_x + total_y)/(2*(len(answer) + len(facit)))
    return final
    
def load_keys():
    
    d1 = 'keywords.json'
    f1 = open(d1)
    corpus = json.load(f1)
    f1.close()
    return corpus['key']
    
def keywords_count(answer, floor = 0.8):
    import ngram
    keys = load_keys()
    
    answer = remove_shit(answer)
    answer = remove_stops(tokenizer(answer))
    answer = tokenizer(answer)
    #print("Clean answer : ", answer)

    G = ngram.NGram(answer) 
    
    candidate = 0
    sum = 0
    for word in keys:
        candidate = G.search(word, threshold=floor)
        if len(candidate) > 0:
            sum = sum + 1
    return sum 
    
def read_corp(query = 'container'):
    
    d1 = 'question11_syn.json'
    f1 = open(d1)
    corpus = json.load(f1)
    f1.close()
    return corpus[query]
    
def corpus_based_sim(answer, floor = 0.8):
    import ngram
    
    #print(read_corp())
    
    facit = "need know amount vinegar container type material time temperature experiment sample rinse distilled water size surface area drying method"
    
    answer = remove_shit(answer)
    answer = remove_stops(tokenizer(answer))
    answer = tokenizer(answer)
    G = ngram.NGram(answer) 
    
    #print("Clean answer : ", answer)
    facit = tokenizer(facit)
    count = 0
    for word in facit :
        candidate = G.search(word, threshold=floor)
        if len(candidate) > 0 :
            count = count + 1
        else :
            syn = read_corp(word)
            for s in syn:
                candidate = G.search(s, threshold=floor)
                if len(candidate) > 0 :
                    count = count + 1
    #print("count = ", count)
    sim1 = count/max(len(facit),24)  
    sim2 = count/max(len(facit),len(answer))  
    return sim1,sim2
    


# answer = question.get_text_by_score(quest=0,score=2)[0]
# print(keywords_count(answer))
# 
# print(corpus_based_sim(answer,0.7))
# print()
# print(partial_words_overlap(answer))
# print()
# print(sem_matrix(answer))
# 
# print("PPPPPPPPPPPPPPPPPPPPPPP å PPPPPPPPPPPPPPPPPPPPPPPPPPPPPP")


    