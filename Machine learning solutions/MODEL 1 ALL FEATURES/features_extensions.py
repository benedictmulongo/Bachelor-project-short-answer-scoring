from nltk import *
import numpy as np
from text_processing import *
from nltk.corpus import wordnet as wn
import json 
import gensim
from gensim import corpora, models, similarities
from BLEU import *
from data import data 
question = data()

class extensions :
    
    def __init__(self):
        #Get alternativ answers 
        doc = 'answers_alternativ_texts.json'
        f1 = open(doc)
        self.references = json.load(f1)['answers']
        f1.close()
        
        #Get the keywords 
        doc = 'keywords.json'
        f2 = open(doc)
        self.keywords = json.load(f2)['key']
        f2.close()
        
        # Train a LSI by the text in the references file
        self.corpus = [tokenizer(doc) for doc in self.references] 
        self.dictionary = corpora.Dictionary(self.corpus)
        self.doc_term_matrix = [self.dictionary.doc2bow(doc) for doc in self.corpus]
        self.lsi = models.LsiModel(self.doc_term_matrix, id2word=self.dictionary, num_topics=50)
        self.index = similarities.MatrixSimilarity(self.lsi[self.doc_term_matrix], num_features=len(self.dictionary))
        
    def LSI(self, ans):

        ans_bow = self.dictionary.doc2bow(tokenizer(ans.lower()))
        ans_lsi = self.lsi[ans_bow]
        #index = similarities.MatrixSimilarity(lsi[doc_term_matrix])
        sims = self.index[ans_lsi]
        vect = list(sims)
        #return sum(vect)/len(vect), max(vect)
        return float(max(vect))

    # compute the overlap between 
    # the candidate answer and all the answers 
    # in the references file to be used 
    # by jaccard and dice similarity
    def overlap_sim(self, ans):
        import ngram
        candidate = tokenizer(ans)
        count = 0
        all_sim = []
        reference = self.references
        for ref in reference :
            G = ngram.NGram(tokenizer(ref))
            for word in candidate :
                match = G.search(word, threshold=0.8)
                if len(match) > 0 :
                    count = count + 1
            all_sim.append((count, len(candidate), len(tokenizer(ref))))
            count = 0
        
        return all_sim
        
    # calculate the jaccard similarity 
    def jaccard_sim(self, ans):
        overlap = extensions.overlap_sim(self,ans)
        score = 0
        norm = len(overlap)
        max = 0
        svar = 0.0001
        facit = 0.0001
        for (meet, x, y) in overlap:
            deno = x + y - 2*meet
            score = score  +  meet/deno
            if meet > max :
                max = meet
                svar = x
                facit = y
        

        #return score, score/norm, max/(svar + facit)
        return max/(svar + facit + 0.000000001)
    # calcualte the dice similarity 
    def dice_sim(self, ans):
        overlap = extensions.overlap_sim(self,ans)
        score = 0
        norm = len(overlap)
        max = 0
        svar = 0
        facit = 0
        for (meet, x, y) in overlap:
            deno = x + y
            score = score  +  (2*meet)/deno
            if meet > max :
                max = meet
                svar = x
                facit = y
        #avoid divide by zero
        if norm == 0 :
            norm = 0.00001
            score = 0
        #return score, score/norm, max/(svar + facit)
        return float(score/(norm + 0.00000000001))
    
    def word_grams(self, words, min=1, max=3):
        
        s = []
        for n in range(min, max):
            for ngram in ngrams(words, n):
                s.append(' '.join(str(i) for i in ngram))
        return s
    
    #Compute the the kewqords coverage rate 
    # in the candidate answer 
    def keywords_normalizer(self, answer):
        import ngram
        keys = self.keywords
        norm = len(keys)

        candidate = tokenizer(answer)
        G = ngram.NGram(extensions.word_grams(self, candidate))
        score = 0
        for keyword in keys :
            match = G.search(keyword, threshold=0.8)
            if len(match) > 0 :
                score = score + 1
        
        return float((score / (norm + 0.000000001))*100)
        
    def bleu_sim(self, ans):        
        score = bleu_similarity(ans, self.references)
        return score
    

