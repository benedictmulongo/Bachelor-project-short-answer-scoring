import numpy as np
from numpy import *
from numpy import zeros
from features import feature_extraction
from text_processing import *
import arff
import json
from data import data

"""
This python file computes all feature for our 
firs model. The computation time is very high. 

You can use the function compute_features_vector()
to generete a file with all features computed 

or you can use the function get_features() 
as X, y = get_features() 
to a genereta a features file and get X, y 
ready to be used in any classifier 
"""
 
question = data()

data = {
    u'attributes':[
        (u'cosine', u'REAL'),
        (u'nKeywords', u'REAL'),
        (u'lsa', u'REAL'),
        (u'partwords', u'REAL'),
        (u'lang_sim', u'REAL'),
        (u'lda', u'REAL'),
        (u'align', u'REAL'),
        (u'corpus_sim', u'REAL'),
        (u'lda_extract', u'REAL'),
        (u'cos_score', u'REAL'),
        (u'bingo_score', u'REAL'),
        (u'jaccard', u'REAL'),
        (u'dice sim', u'REAL'),
        (u'keywords_norm', u'REAL'),
        (u'LSI_query', u'REAL'),
        (u'bleuscore', u'REAL'),
        (u'keyf_score', u'REAL'),
        (u'fgram_score ', u'REAL'),
        (u'holo_score', u'REAL'),
        (u'category',[u'0', u'1', u'2',u'3'])],
    u'data': [
    ],
    u'description': u'',
    u'relation': u'short answer attributes'
}


def compute_features_vector(file = 'FEATURES_test.json', answers = question.get_question(0)):
    
    
    for index, answer in enumerate(answers) :
        score = answer['score_1']
        text = answer['text']
        text = remove_stops(tokenizer(text))
        len1 = len(text)
        len2 = len(tokenizer(text))
        
        feature = feature_extraction(text)
        
        cosin = feature.cosine_similarity()
        keywords = feature.keywords_similarity()
        _,_,_,LSA = feature.lsa_similarity()
        part_words,_ = feature.partial_words_similarity()
        language = feature.language_similirity()
        lda = feature.lda_similarity()
        al1, al2 = feature.align_similarity()
        c1,c2 = feature.corpus_similarity()
    
        lda_ext = feature.lda_extract()
        cos_sc = feature.cos_score()
        bingo_sc = feature.bingo_score()
        
        jacc = feature.jaccard()
        dic_sim = feature.dice()
        keys_norm = feature.keywords_norm()
        lsi_query = feature.LSI_query()
        bleu_score = feature.bleuscore()
        keyf = feature.keyf_score()
        fgram  = feature.fgram_score()
        holo = feature.holo_score()
    
    
        info = [cosin,keywords,LSA,part_words,language,lda,al1,c1,lda_ext, cos_sc,bingo_sc, jacc,dic_sim,keys_norm,lsi_query, bleu_score,keyf ,fgram, holo , score]
        print(info)
        info = [float(x) for x in info]
        
        data['data'].append(info)
        f = open(file, 'w')
        json.dump(data, f, indent=2)
        f.close()
        print("Index = ", index)
    
    # write features to file 
    f = open(file, 'w')
    json.dump(data, f, indent=2)
    f.close()
    
def list_of_list_to_int(s):
    return np.array([[float(y) for y in x] for x in s])
  
def list_to_int(s):
    return np.array([int(x) for x in s])

def read_feat():
    
    d = 'FEATURES_test.json'
    f = open(d)
    filen = json.load(f)
    f.close()
    
    return filen['data']

def get_features(f = 'FEATURES_test.json'):
    # Compute all features
    compute_features_vector(file = f)
    # Get the data X and the Labels Y 
    # of the computed features
    respons = np.array(read_feat())
    y = list_to_int(respons[:,-1])
    X = list_of_list_to_int(respons[:,0:col-1])
    return X, y
    
    
def vectors_to_arff_format():
    # To be used for Weka analysis
    data['data'] = read_feat()
    print(arff.dumps(data))

    
# compute_features_vector()