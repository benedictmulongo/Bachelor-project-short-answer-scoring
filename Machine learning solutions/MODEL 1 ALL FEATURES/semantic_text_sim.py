""""
This is an implementation of the algorithms describe in the following paper 
Semantic Similarity of Short Texts
Aminul Islam, Diana Inkpen, 2007
available on : https://www.semanticscholar.org/paper/Semantic-Similarity-of-Short-Texts-Islam/082ee254cf5444639ed43633ebcd275de381639b?tab=abstract

"""
from LCS import compute_LCS
import math
from nltk import *
from nltk.util import ngrams
import numpy as np
from text_processing import *
from semantic_similarity import semantic

def string2Char(s):
 
    k = [''.join(x+ " ") for x in s]
    k = ''.join(k).strip()
    return k

def word_grams(words, min=1, max=4):
    
    s = []
    for n in range(min, max):
        for ngram in ngrams(words, n):
            s.append(''.join(str(i) for i in ngram))
    return s
    
def string_ngrams(s):
    
    length = len(s)
    ret = string2Char(s).split(" ") 
    ng = word_grams(ret, min = 1, max = length)
    return ng 
    
def mclcs_1(string1, string2):
    
    stringMax = string1.strip()
    stringMin = string2.strip()
    
    r = len(stringMin)
    s = len(stringMax)
    
    if r > s :
        temp = r
        r = s
        s = temp
        
        tmp = stringMin
        stringMin = stringMax
        stringMax = tmp
    
    while len(stringMin) >= 0 :
        
        if stringMin in stringMax :
            break
        else :
            stringMin = stringMin[0:len(stringMin)-1]

    return stringMin
    
def max_ngram(ng):
    
    #copy list ng to ng_b
    ng_b = list(ng)
    ng_c = list(ng)
    x = [len(x) for x in ng]
    element = 0
    if x == [] :
        element = ''
    else :
        index = np.argmax(x)
        element = ng[index]
        ng_c.remove(element)
        
    return element,ng_c,ng_b
    
def mclcs_n(string1, string2):
    
    stringMax = string1.strip()
    stringMin = string2.strip()
    
    r = len(stringMin)
    s = len(stringMax)
    
    if r > s :
        temp = r
        r = s
        s = temp
        
        tmp = stringMin
        stringMin = stringMax
        stringMax = tmp
    back = 0
    # determine all n-grams from stringMin where
    # n = 1 ... |stringMin| and r_hat is the set 
    # of n-grams 
    # r_hat = string_ngrams(stringMin)
    r_hat = string_ngrams(stringMin)
    while len(stringMin) >= 0 :
        
        # x = {x|x C r_hat, x = Max(r_hat)}
        x,ng_mod,ng_original = max_ngram(r_hat)
        
        if x in stringMax :
            back = x
            break
        else :
            r_hat = ng_mod
            stringMin = x

    return back

    
def word_grams_source(words, min=1, max=4):
    s = []
    for n in range(min, max):
        for ngram in ngrams(words, n):
            s.append(' '.join(str(i) for i in ngram))
    return s

def string_similarity(string_a,string_b, w1 = 0.33, w2 = 0.33 , w3 = 0.33):
    
    a_len = len(string_a)
    b_len = len(string_b)
    
    lcs = compute_LCS(string_a,string_b)
    mclcs1 = mclcs_1(string_a,string_b)
    mclcsN = mclcs_n(string_a,string_b)
    
    NLCS = (len(lcs))**2/(a_len*b_len)
    NMLCLCS = (len(mclcs1))**2/(a_len*b_len)
    NMLCLCS_N = (len(mclcsN))**2/(a_len*b_len)
    
    result = NLCS*w1 + NMLCLCS*w2 + NMLCLCS_N*w3
    
    return result
    
def even(x):
    if x % 2 == 0 :
        return True
    else :
        return False
        
    
def common_word_order_sim(text_a, text_b):
    stringMax = text_a
    stringMin = text_b
    a_max = tokenizer(text_a.strip())
    b_min = tokenizer(text_b.strip())  
    
    
    r = len(b_min)
    s = len(a_max)
    
    if r > s :
        temp = r
        r = s
        s = temp
        
        tmp = stringMin
        stringMin = stringMax
        stringMax = tmp
        
        a_max = tokenizer(stringMax)
        b_min = tokenizer(stringMin)

    A = []    
    for x in a_max:
        A.append(x)
    B = []
    for y in b_min:
        B.append(y)

    max_common = []
    max_common_index = []
    max_not_com = []
    i = 1
    for x in A:
        if (x in B) and (x not in max_common):
            max_common.append(x)

            max_common_index.append(i)
            i = i + 1
        else :
            max_not_com.append(x)
    max_not_com = ' '.join(max_not_com)
            
    match = len(max_common)
    
    min_common = []
    min_not_com = []
    for x in B:
        if x in max_common:
            min_common.append(x)
        else :
            min_not_com.append(x)
    min_not_com = ' '.join(min_not_com)

    max_common = remove_duplicates(max_common)

    min_common_index = []      
    for x in min_common:
        ind = max_common.index(x) + 1
        min_common_index.append(ind)

    a = np.array(max_common_index)
    b = np.array(min_common_index)
    norm = np.linalg.norm((a - b), ord=1)
    
    back = 1
    if even(match):
        back = 1
        back = back - ((2*norm)/(match**2))
    elif even(match) == False and match > 1 :
        back = 1
        back = back - ((2*norm)/(match**2 - 1))   
    elif even(match) == False and match == 1 :
        back = 1
    
    return stringMin, stringMax, r,s, max_common,min_common,max_not_com,min_not_com,back
    
def norm_semsim(r,s,alpha = 1):
    sem = semantic()
    v = sem.word_sim_rel(s,r)
    if v > alpha:
        v = 1
    else :
        v = v/alpha
        
    return v
    
def joint_matrix(alphas, betas, param1 = 0.2, param2 = 0.8):
    
    # Transformation of the matrices 
    # to numpy arrays
    matrix_a = np.array(alphas)
    matrix_b = np.array(betas)
    
    # Scalar multiplication of the matrices 
    # with the scaling factor 
    matrix_a = np.multiply(matrix_a, param1)
    matrix_b = np.multiply(matrix_b, param2)
    
    # Perfom matrix_a + matrix_b
    matrix_joint = np.add(matrix_a, matrix_b)
    return matrix_joint
    
    
def find_max(matrix):
    #not so effective but the only solution right now...
    max = -float('inf')
    r = 0
    c = 0
    if len(matrix) == 0 :
        max = 0
        r = -1
        c = -1
    else :
        for row in range(len(matrix)): 
            for column in range(len(matrix[row])) :
                if matrix[row][column] > max:
                    max = matrix[row][column]
                    r = row
                    c = column
    return max,r,c
    
def max_value_reduction(matrix, length):
    
    matrix = matrix
    bag = []
    y_ij = float('inf')
    var = length - len(bag)
    
    while y_ij !=  0 or var != 0 or (y_ij != 0 and var != 0) :
        max_value,row,col = find_max(matrix)
        if row == -1 and col == -1 :
            break 
        reduc = np.delete(matrix,row,0)
        matrix = np.delete(reduc,col,1) 
        
        y_ij = max_value 
        if max_value >= 0:
            bag.append(max_value)

        var = length - len(bag)

    return bag
    
def compute_similarity(r,s, w = 0.2):
    
    text_a = remove_stops(tokenizer(r.lower()))
    text_b = remove_stops(tokenizer(s.lower()))
    text_a = lemmatizer(text_a)
    text_b = lemmatizer(text_b)
    
    _,_, min, max, l1 , l2,str_max,str_min, score = common_word_order_sim(text_a,text_b)
    #stringMin, stringMax, r,s, max_common,min_common,max_not_com,min_not_com,back
    
    min_tokens = tokenizer(str_min)
    max_tokens = tokenizer(str_max)
    matrix_alpha = []
    matrix_beta = []
    mat_joint = []
    # if all term matches go to step 6
    if abs(len(l1) - min) == 0 :
        #step 6
        k = 0
    else :
        infinity = -float('inf')
        row = min - len(l1) 
        col = max - len(l1)
        matrix_alpha = [[0] * col for _ in range(row)]
        for i,x in enumerate(min_tokens):
            for j,y in enumerate(max_tokens):
               alpha = string_similarity(x,y)
               matrix_alpha[i][j] = alpha
               
        matrix_beta = [[0] * col for _ in range(row)]
        for i,x in enumerate(min_tokens):
            for j,y in enumerate(max_tokens):
               beta = norm_semsim(x,y)
               if beta == infinity:
                   beta = 0
               matrix_beta[i][j] = beta
        mat_joint = joint_matrix(matrix_alpha, matrix_beta)
    
    Delta = len(l1)
    S_zero = score 
    reduc = max_value_reduction(mat_joint, len(mat_joint))
    p = sum(reduc)
    m = len(tokenizer(r))
    n = len(tokenizer(s))

    
    F_score = (Delta*(1 - w + w*S_zero) + p) * (m + n)
    F_score = F_score /(m*n) 
    
    return F_score
 
def sem_compute_similarity(answer):
    
    facit_text = "amount quantity vinegar container cup type sort vinegar container sort type materials test size surface area material materials should used long time minute minutes sample rinsed rinse distilled distiled water dry drying method use size type container cup temperature experiment"
    
    return compute_similarity(answer,facit_text)

# a = "we need to know how much vinegar to put in the container"
# b = "we require to recognize the quantity of vinegar in cup"
# facit_text = "amount quantity vinegar container cup type sort vinegar container sort type materials test size surface area material materials should used long time minute minutes sample rinsed rinse distilled distiled water dry drying method use size type container cup temperature experiment"
# print(sem_compute_similarity(b))
# print("Sim w = 0.2 -> ", compute_similarity(a,facit_text))
