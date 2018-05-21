
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
import math
import json
import numpy as np
from text_processing import *
from gensim import corpora
from gensim import *

Lda = gensim.models.ldamodel.LdaModel
dictionary = corpora.Dictionary.load('lda_model_dictionary.dict')
model = Lda.load('lda_model.gensim', mmap='r')

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
    
def lda_features(answer):
    ans = clean(answer)
    ans = dictionary.doc2bow(tokenizer(ans))
    svar = model[ans]
    length = len(svar)
    sum = 0
    if length <= 3 :
        for (x,y) in svar[0:length]:
            sum = sum + y
    else :
        sum = length
    
    #return length, svar, svar[0:demi], sum 
    return sum 
    
# from data import data 
# question = data()
# answer = question.get_text_by_score(quest=0,score=3)[10]
# 
# print(lda_features(answer))




