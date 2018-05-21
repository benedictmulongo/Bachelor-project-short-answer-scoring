"""BLEU score implementation."""
import math
from collections import Counter
from nltk import *
import numpy as np
from text_processing import *
from nltk.corpus import wordnet as wn
import json 
from data import data 
import ngram
from nltk.util import ngrams

question = data()

def n_grams(words, min=1, max=3):
    
    s = []
    all = []
    for n in range(min, max):
        for ngram in ngrams(words, n):
            s.append(' '.join(str(i) for i in ngram))
        all.append(s)
        s = []
    return all

def occurence_count(word, references_grams):
    list = []
    score = 0
    for i, ref in enumerate(references_grams):
        result = ref.search(word, threshold=0.8)
        if len(result) > 0:
            score = len(result)
        else :
            score = 0
        list.append(score)
    return list, max(list), np.argmax(list)
        

def M_bleu(candidate, references_grams):
    
    candidate_tokens = tokenizer(candidate)
    lenghtCandidate = len(candidate_tokens)
    #print("Cand = ", candidate_tokens , " len = ", lenghtCandidate)
    candidate_grams = n_grams(candidate_tokens)

    all_precision = []
    for i, p in enumerate(candidate_grams) :
        occurence = []
        sum = 0
        for word in p:
            # First modification - allowing partial word match 
            # instead of using exact match between words
            _,max_ref,index = occurence_count(word, references_grams)
            candidate_text = ngram.NGram(candidate_grams[i])
            word_occur = candidate_text.search(word, threshold=0.9)
            count = len(word_occur)
            count_p_clipped = min(count, max_ref) 
            sum = sum  + count_p_clipped
        #print("sum = ", sum)
        MUP = sum / (lenghtCandidate + 0.0000001)
        all_precision.append(MUP)
        sum = 0
    # Second modification when calculating the combined
    # MUP all_precision use weighting to set the 
    # importance of the given n-grams 
    # weight = [0.5,0.25,0.125,0.125]
    # weight = [0.75,0.083,0.083,0.083]
    #print()
    MUP = np.array(all_precision)
    #print("MUP before = ", MUP)
    MUP = np.where(MUP != 0.0 , MUP, np.finfo(float).eps)
    #print("MUP after = ", MUP)
    MUP_log = np.log(MUP)
    #print("MUP log = ", MUP_log)
    
    w = np.array([0.85,0.15])
    combined_MUP = np.dot(MUP_log, w)
    return combined_MUP, math.exp(combined_MUP)
    
def bleu(answer, references, theta = 0.0001):
    
    references_grams = []
    result = []
    for reff in references :
        G = ngram.NGram(tokenizer(reff))
        _, expp = M_bleu(answer, [G])
        result.append(expp)
        
    maxScore = max(result)
    maxIndex =  np.argmax(result)
    refMax = []
    for index, refs in enumerate(result) :
        if abs(refs - maxScore) < theta :
            refLength = len(tokenizer(references[index]))
            candLength = len(tokenizer(answer))
            refMax.append((refs, index, brevity_penalty(candLength, refLength)))
            
    # Find the selected reference answer 
    SRA_m_bleu = 0
    SRA_bp = 0  
    SRA_index = 0      
    for (m_bleu, index, bp) in refMax:
        if bp > SRA_bp :
            SRA_bp = bp
            SRA_index = index
            SRA_m_bleu = m_bleu 
    #print((SRA_m_bleu, SRA_bp, SRA_index))
    
    #return maxScore, maxIndex, refMax
    return SRA_m_bleu, SRA_bp, index
    
def brevity_penalty(candidateLength, referenceLength):
    back = 1
    if candidateLength > referenceLength :
        back = 1
    else :
        back = math.exp(1 - (referenceLength / (candidateLength + 0.000001) ))
    return back 
    
def read_open():
    doc = 'answers_alternativ_texts.json'
    f1 = open(doc)
    ref = json.load(f1)['answers']
    f1.close()
    return ref
    
def even(x):
    if x % 2 == 0 :
        return True
    else :
        return False
        

def common_word_order(P,R):
    
    # Assume that R > P
    p = tokenizer(P)
    r = tokenizer(R)
    # if not 
    if len(p) > len(r):
        temp = p
        p = r
        r = temp
    # find common word 
    p_set = set(p)
    r_set = set(r)
    p_set_diff = p_set.difference(r_set)
    common = list(p_set - p_set_diff)
    
    #
    com_p = []
    com_r = []
    for word in common :
        index_p = p.index(word)
        com_p.append((word, index_p))
        com_p = sorted(com_p, key=lambda x: x[1])
        
        index_r = r.index(word)
        com_r.append((word, index_r))
        com_r = sorted(com_r, key=lambda x: x[1])
    

    for i, (x,y) in enumerate(com_p):
        com_p[i] = (x, i+1)
        for j, (w,z) in enumerate(com_r):
            if w == x :
                com_r[j] = (w,com_p[i][1])

    x = np.array([ w for (z,w) in com_p])
    y = np.array([ w for (z,w) in com_r])
    norm = np.linalg.norm((x - y), ord=1)
    alpha = len(common)

    S_0 = 1
    if even(alpha):
        S_0 = 1
        S_0 = S_0 - ((2*norm)/(alpha**2 + 0.000001))
    elif even(alpha) == False and alpha > 1 :
        S_0 = 1
        S_0 = S_0 - ((2*norm)/(alpha**2 - 1 + 0.000001))   
    elif even(alpha) == False and alpha == 1 :
        S_0 = 1

    return S_0
    

def bleu_similarity(answer, references, lambdar = 0.85):
    
    m_blue_score, bp_score, index = bleu(answer,references)
    s_0 = common_word_order(answer, references[index])
    sim = lambdar*m_blue_score*bp_score + (1-lambdar)*s_0
    
    return float(sim) 
    

