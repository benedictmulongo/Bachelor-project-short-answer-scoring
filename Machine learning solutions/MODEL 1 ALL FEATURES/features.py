from text_processing import *
import json
from nltk.corpus import wordnet as wn
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob as tb
# Import file 
from cosine_similarity import *
from alignment import *
from language_model import *
from lda_features import *
from LSA import *
from short_sentence_similarity import *
from simple_features import *
from features_extensions import *
from lda_extraction import lda 
from cosim import cos_give_score
from bingo import bingo_give_score
from keyword_f import Keyword
from ngram_f import Fgram
from holistic import Holistic


class feature_extraction :
    
    def __init__(self, ans):
        self.answer = ans
        self.facit = "amount quantity vinegar container cup type sort vinegar container sort type materials test size surface area material materials should used long time minute minutes sample rinsed rinse distilled distiled water dry drying method use size type container cup temperature experiment"
        self.facit_sem = "amount quantity vinegar container cup type sort vinegar container sort type test size surface area material long time minute rinse distilled distiled water dry drying method temperature experiment"
        self.ext = extensions()
        self.extract = lda()
        
        self.keyf = Keyword()
        self.fgram = Fgram()
        self.holo = Holistic()

    def lda_extract(self):
        return self.extract.give_score(self.answer)
    def cos_score(self):
        return cos_give_score(self.answer)
    def bingo_score(self):
        return bingo_give_score(self.answer)
    def keyf_score(self):
        return self.keyf.give_score(self.answer)
    def fgram_score(self):
        return self.fgram.give_score(self.answer)
    def holo_score(self):
        return self.holo.give_score(self.answer)  

    def jaccard(self):
        return self.ext.jaccard_sim(self.answer)
        
    def dice(self):
        return self.ext.dice_sim(self.answer)
        
    def keywords_norm(self):
        return self.ext.keywords_normalizer(self.answer)
        
    def LSI_query(self):
        return self.ext.LSI(self.answer)
        
    def bleuscore(self):
        return self.ext.bleu_sim(self.answer)

    def cosine_similarity(self):
        return compute_cosine(self.answer)
        
    def align_similarity(self):
        return compute_align_all(self.answer)
        
    def language_similirity(self):
        return compute_perp(self.answer)
        
    def lda_similarity(self):
        return lda_features(self.answer)
        
    def lsa_similarity(self):
        return LSA_predict(self.answer)
        
    def partial_words_similarity(self):
        return partial_words_overlap(self.answer)
   
    def keywords_similarity(self):
        return keywords_count(self.answer)
        
    def corpus_similarity(self):
        return corpus_based_sim(self.answer)
        