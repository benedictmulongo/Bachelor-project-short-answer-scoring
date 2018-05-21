from data import data 
from rake_nltk import Rake
import nltk
from textblob import TextBlob as tb
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
import json
from text_processing import *
import nltk
import numpy as np
from scipy.special import logsumexp
from data import data 
question = data()

def language(docs='0', ind = 0):
    # Dictionary - synomyns
    lang = 0
    if ind == 0 :
        lang = 'LANGUAGE_MODEL_FACITT.json'
    else:
        lang = 'LANGUAGE_MODEL_FACIT.json'
    model = open(lang)
    lingua= json.load(model)
    model.close()
    return lingua[docs]

def term_docs_prob(text, term):
    occurence = text.count(str(term))
    length = len(tokenizer(text))
    return (occurence + 1)/(length + 1)
    
def add_docs(corpus):
    ret = ""
    for docs in corpus:
        ret = ret + docs + " "
    return ret.strip()
    
    
#corpus = [doc1, doc2, doc3 ........]
def linear_inter(corpus, query, alpha):
    query = tokenizer(query.lower())
    Mc = add_docs(corpus)
    #total = 1
    total = []
    for doc in corpus:
        prob_d = []
        for word in query:
            
            Pmc = term_docs_prob(Mc,word)
            Pmd = term_docs_prob(doc,word)
            
            Pmc_alpha =(1 - alpha)*Pmc 
            Pmd_alpha = (alpha)*Pmd
            sum = Pmd_alpha + Pmc_alpha
            prob_d.append(sum)
        total.append(np.prod(prob_d))
    return total
   
def linear_inter_log(corpus, query, alpha):
    query = tokenizer(query.lower())
    Mc = add_docs(corpus)
    #total = 1
    total = []
    upper_bound = []
    for doc in corpus:
        prob_d = []
        for word in query:
            
            Pmc = term_docs_prob(Mc,word)
            Pmd = term_docs_prob(doc,word)
            
            Pmc_alpha =(1 - alpha)*Pmc 
            Pmd_alpha = (alpha)*Pmd
            sum = Pmd_alpha + Pmc_alpha
            #print("Sum = ", sum, " Query = ", word)
            prob_d.append(sum)
        total.append(logsumexp(prob_d))
    #print("Upper bound = ",upper_bound)
    return total
    
def calcul_perplexity(ans,corp,w=[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125],a=0.5):
    
    ans = keep_text(correct_singularise(remove_stops(tokenizer(clean(ans.lower())))))
    length = len(tokenizer(ans))
    d = linear_inter(corp, ans, a)
    array = np.array(d)
    factor = 0
    if length > 0 :
        prob = np.power(array,1/length)
        prob = prob*100
        factor = np.dot(prob,np.array(w))
        
    return factor
 
def compute_perp(answer):
    
    lang_0 = language('0')
    lang_1 = language('1')
    lang_2 = language('2')
    lang_3 = language('3')
    lang_4 = language('4')
    lang_5 = language('5')
    lang_6 = language('6')
    lang_7 = language('7')
    
    corpus = [lang_0,lang_1,lang_2,lang_3,lang_4,lang_5,lang_6,lang_7]
    
    # corpus1 = ["amount quantity vinegar acid cup kettle jar was used in each container", "need know type sort vinegar acid was used in each container cup", "need know sort type materials material to test", "need know size surface area material materials should used", "know long time minute min minutes each sample rinsed rinse in distilled water liquid", "drying procedure approach method use used","size type container cup jar used", "know temperature experiment"]
    
    return calcul_perplexity(answer,corpus)



# print(compute_perp("We need to know quantity vinegar"))
