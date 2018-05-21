
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
import math
import json
import numpy as np
from text_processing import *
from gensim import corpora

bad = "An experiment is a procedure carried out to support, refute, or validate a hypothesis. Experiments provide insight into cause and effect by demonstrating what outcome occurs when a particular factor is manipulated. Experiments vary greatly in goal and scale, but always rely on repeatable procedure and logical analysis of the results."

corpus = ["amount quantity vinegar acid cup kettle jar was used in each container", "need know type sort vinegar acid was used in each container cup", "need know sort type materials material to test", "need know size surface area material materials should used", "know long time minute min minutes each sample rinsed rinse in distilled water liquid", "drying procedure approach method use used","size type container cup jar used", "know temperature experiment", bad]


stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in corpus] 
#print(doc_clean)
#print()
dictionary = corpora.Dictionary(doc_clean)
dictionary.save('lda_model_dictionary.dict')
#print(dictionary)
#print()
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
#print(doc_term_matrix)
Lda = gensim.models.ldamodel.LdaModel
model = Lda(doc_term_matrix, num_topics=9, id2word = dictionary, passes=100)
model.save('lda_model.gensim')

#print(model.print_topics(num_topics=8, num_words=5))

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
    
from data import data 
question = data()
answer = question.get_text_by_score(quest=0,score=0)[5]

print(lda_features(answer))

