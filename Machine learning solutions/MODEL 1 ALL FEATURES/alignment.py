from nltk import *
import numpy as np
from text_processing import *
from nltk.corpus import wordnet as wn
import json 
from data import data 

question = data()

def read_corp():
    
    d1 = 'question11_syn.json'
    f1 = open(d1)
    corpus = json.load(f1)
    f1.close()
    return corpus
    
def corpus_based_alignment(answer, facit):
    
    # add noun - noun comparison and verb - verb 
    answer = keep_text(answer.lower())
    answer = remove_stops(remove_duplicates(tokenizer(answer)))
    answer = list(tokenizer(answer))
    
    facit= keep_text(facit.lower())
    facit = remove_stops(remove_duplicates(tokenizer(facit)))
    facit = list(tokenizer(facit))
    
    fac_len = len(facit)
    ans_len = len(answer)
    #print("FAcit = ", facit)
    #print("Answer = ", answer)
    corp = read_corp()
    import ngram
    G = ngram.NGram(answer) 
    align = set()
    for word in facit :
        cand = G.search(word, threshold=0.9)
        if len(cand) > 0:
            align.add(word)
        else :

            for ord in answer:
                syn = corp[word]
                for s in syn :
                    cand = G.search(s, threshold=0.9)
                    if len(cand) > 0:
                        align.add(word)
                        #answer.remove(ord)
                        break

    score = len(align)
    score1 = (2*score)/(fac_len + ans_len)
    score2 = (2*score)/(fac_len + 24)
    #return list(align), score,fac_len,ans_len
    return score1,score2

def api_based_alignment(answer, facit):
    from semantic_similarity import semantic
    sem = semantic()
    # add noun - noun comparison and verb - verb 
    answer = keep_text(answer.lower())
    answer = remove_stops(remove_duplicates(tokenizer(answer)))
    answer = list(tokenizer(answer))
    
    facit= keep_text(facit.lower())
    facit = remove_stops(remove_duplicates(tokenizer(facit)))
    facit = list(tokenizer(facit))

    
    fac_len = len(facit)
    ans_len = len(answer)
    # print("FAcit = ", facit)
    # print("Answer = ", answer)
    import ngram
    G = ngram.NGram(answer) 
    align = []
    for word in facit :
        cand = G.search(word, threshold=0.9)
        if len(cand) > 0:
            align.append(word)
        else :
            for ord in answer:
                sim = sem.word_sim_rel(word,ord)
                if sim > 0.6 :
                    print("**")
                    align.append([word,ord])
                    answer.remove(ord)
                    break
                    
    score = len(align)
    score = (2*score)/(fac_len + ans_len)
    #return align, score,fac_len,ans_len
    return score
    
def compute_align(answer, facits):
    """
    does not really work
    """
    cands = sent_tokenizer(answer)
    scores = []
            
    for facit in facits:
        for cand in cands:
            print("***** Aling *****")
            list_a, score_a,fac_len,ans_len =corpus_based_aligment(cand, facit)

   #           score =(2*score_a)/(fac_len + ans_len)
            scores.append(scores)
            
    return np.average(np.array(scores))
   
def compute_align_api(answer, facits):
    """
    does not really work ???
    """
    from semantic_similarity import semantic
    sem = semantic()
    cands = sent_tokenizer(answer)
    scores = []
    for facit in facits:
        for cand in cands:
            print("***** Aling *****")
            list_a, score_a,fac_len,ans_len =api_based_alignment(cand, facit)
            #list_b, score_b,_,_ = api_based_alignment(facit,cand)

            score =(2*score_a)/(fac_len + ans_len)
            scores.append(scores)
            
    return np.average(np.array(scores))

    
def compute_align_all(answer):
    facit = "need know quantity vinegar used container need know type sort vinegar used container need know sort type material test need know size surface area material used know long time sample rinse distilled water need know drying method use know size type container use know temperature experiment"
    
    return corpus_based_alignment(answer, facit)
    
# answer = question.get_text_by_score(quest=0,score=3)
# 
# facit = "need know quantity vinegar used container need know type sort vinegar used container need know sort type material test need know size surface area material used know long time sample rinse distilled water need know drying method use know size type container use know temperature experiment"
# for x in answer:
#     print(corpus_based_alignment(x, facit))
# print(api_based_alignment(answer, facit) )
