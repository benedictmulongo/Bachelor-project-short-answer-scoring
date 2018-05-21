import numpy as np
from numpy import *
from numpy import zeros
from textblob import TextBlob as tb
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
#SVD
from scipy.linalg import svd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as cosine
from data import data 
question = data()


def flatten(s):
    flt = [y for x in s for y in x]
    return flt
    
def filter(s):
    k = [flatten([ re.split(';|,|\*|\n|\.|\/|',y) for y in x ]) for x in s ]
    return k

def sentences_finder(text):
    # Comma splitter
    mass = text.split(',')
    length = len(mass)
    
    #sentence finder blob 
    blob = tb(text)
    # grammatical correction
    blob = blob.correct()
    #***********************
    sent = blob.sentences
    ret = [ ''.join(x).strip() for x in sent]
    if length >= 4:
        ret = mass
    return ret

def tokenizer(text):
    return word_tokenize(text)

def remove_stops(text):

    stop = ['.',')','(','/','=','*','¨','%','€','|','&',',','also','would','?','.^']
    result = ""
    for x in text:
        if (x in stopwords.words('english')) != True and (x in stop) != True and len(x) > 2:
            result = result + " " + x
    return result

def clean_paragraph(text1, found_sent = True):
    X = text1
    if found_sent:
        X = sentences_finder(text1)
    x_lower = [x.lower() for x in X]
    x_tokens = [tokenizer(x) for x in x_lower]
    x_clean = [tokenizer(remove_stops(x)) for x in x_tokens]
    return x_clean

def remove_duplicates(values):
    output = []
    seen = set()
    for value in values:
        # If value has not been encountered yet,
        # ... add it to both list and set.
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output
    
def duplicat(array):
    x = [ remove_duplicates(x) for x in array]
    return x

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else : 
            c = 0
    return returnVec

def chunk_matrix(matrix, y = 0, n = 0):
    k = np.array(matrix)
    ut = k[0:y,0:n]
    return ut
    
def svd_chunk(matrix, chunk = False, index = 0):
    S,K,U = svd(matrix)
    row, col = np.shape(matrix)
    if chunk:
        S = chunk_matrix(S,row,index)
        K = chunk_matrix(np.diag(K),index,index)
        U = chunk_matrix(U,index,row)
    return S,K,U
    
def term_doc(matrix, chunk = False ,index = 0):
    S,K,U = svd_chunk(matrix, chunk, index)
    terms = np.matmul(S,K)
    document = np.matmul(K,U)
    return terms, document, S, K, U

def split_xy(matrix):
    N,skit = np.shape(matrix)
    x = [i[0] for i in matrix]
    y = [i[1] for i in matrix]
    return x,y,N
    
def split_coord(matrix):
    sit, N = np.shape(matrix)
    list = []
    y = []
    tmp = []
    for i in range(N):
        for j in range(sit):
            tmp.append(matrix[j][i])
        list.append(tmp)
        tmp = []
    return list
    
def similarity(x,y):
    k = split_coord(x)
    rt = []
    for i in k :
        rt.append(cosine(np.array(i).reshape(1,-1), np.array(y).reshape(1,-1))[0][0])
    return rt
    
def query(U,K,V, key):
    Uq = np.transpose(U)
    Kinv = np.linalg.inv(K)
    result = np.matmul(np.matmul(Kinv, Uq),key)
    return result 
   
def count(digit, list):
    c = 0
    for x in list :
        if x == digit:
            c = c + 1
    return c, len(list), c/len(list)

def train_LSA(myVocabList, ref, trunc = 6 ):
    """
    This function train an LSA model on the reference 
    answer keywords and return b - the vector document 
    a - the term document 
    
    and the matrix S, K, U will can be used to reduce 
    a new query in the same dimension as the trained 
    LSA model 
    """
    count_matrix = []
    
    for x in ref :
        #build count matrix
        count_matrix.append(setOfWords2Vec(myVocabList, x))
    matrix = np.array(count_matrix)
    matrix_term = matrix.transpose()
    #Compute SVD and truncat the result 
    # with trunc = 6 as default 
    a, b, S, K,U = term_doc(matrix_term,True,trunc)
    
    return a, b, S, K, U


def predict_LSA(ans, myVocabList, ref, trunc = 6):
    """
    This function uses train_LSA to train a model 
    and uses further the trained model to reduce 
    the answer in the same dimension and compute 
    similarity between every document in the 
    reference answer and the query (answer) document 
    """
    # Library to find the k-largest element 
    # in a list 
    from heapq import nlargest
    
    # Train LSA 
    # save the model for higher computation time 
    a, b, S, K, U = train_LSA(myVocabList, ref, trunc)
    
    # Take the answer text, clean it 
    # Transform it to a list of list 
    # there every list is considered 
    # as a document to be compare with 
    # the trained LSA model reference
    # answers
     
    answers = clean_paragraph(ans)
    answers = duplicat(answers)
    answers = filter(answers)

    total_4 = 0
    total_6 = 0
    total_all = 0
    for x in answers:

        key = setOfWords2Vec(myVocabList, x)
        result = query(S,K,U,key)
        sim = similarity(b,result)
        # find the 4 largest elements
        # sum them up 
        largest = nlargest(4,sim)
        total_4 = total_4 + sum(largest)
        # sum up everything
        total_all = total_all + sum(sim)
        # Find the 6 largest 
        largest = nlargest(6,sim)
        total_6 = total_6 + sum(largest)
        
    feat = (total_all + total_6 + total_4) /3

    return total_all,total_6, total_4, feat

def LSA_predict(ans, trunc = 6):
    myVocabList = ['need', 'know', 'much', 'quantity', 'vinegar', 'used', 'container', 'type', 'sort', 'materials', 'test', 'size', 'surface', 'area', 'material', 'long', 'time', 'sample', 'rinsed', 'rinse', 'distilled', 'water', 'drying', 'method', 'use','cup','temperature','experiment']
    
    ref = [['need', 'know', 'much', 'quantity', 'vinegar', 'used', 'container', 'cup'], ['need', 'know', 'type', 'sort', 'vinegar', 'used', 'container','cup'], ['need', 'know', 'sort', 'type', 'materials','material' ,'test'], ['need', 'know', 'size', 'surface', 'area', 'material', 'materials', 'used'], ['know', 'long', 'time', 'sample', 'rinsed', 'rinse', 'distilled', 'water','liquid'], ['need', 'know', 'drying', 'method', 'use'], ['know', 'size', 'type', 'container', 'use'],['degree','temperature', 'experiment']]
    
    return predict_LSA(ans, myVocabList, ref, trunc)
  

# print(LSA_predict("I want vinegar vinegar pour pour !"))
# usage : use function predict 
# where the first argument is 
# the input answer and the second
# optional argument is the dimension
# reduction parameter - how much you
# want to reduce the trained model 
# trunc = 6 by default 
# obs trunc = [2,8] otherwise you
# will get error 
# example = predict(ans)

# Test 
# all_text = question.get_text_by_score(quest=0,score=2)[0:5]
# 
# for ans in all_text:
#     print(predict(ans))
#     print()




