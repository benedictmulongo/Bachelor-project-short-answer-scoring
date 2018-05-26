
import json
from nltk.corpus import wordnet as wn
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob as tb

class dictionary :
    
    def __init__(self):
        # Dictionary - synomyns
        self.d = 'dico.json'
        self.f1 = open(self.d)
        self.dico = json.load(self.f1)
        self.f1.close()
        
        #facit 
        self.fac = 'facit.json'
        self.f2 = open(self.fac)
        self.facit = json.load(self.f2)
        self.f2.close()
        
        #world docs 
        self.doc = 'docs.json'
        self.f3 = open(self.doc)
        self.docs = json.load(self.f3)
        self.f3.close()
        
        self.reference = 'You need to know how much vinegar was used in each container. You need to know what type of vinegar was used in each container. You need to know what materials to test. You need to know what size/surface area of materials should be used. You need to know how long each sample was rinsed in distilled water. You need to know what drying method to use. You need to know what size/type of container to use.'
    # NLTK - synomyns 
    def synomyns(self, string):
        str = wn.synsets(string)
        s = [x.name() for x in str]
        s = [x for x in s if string in x]
        s = [[y.name() for y in wn.synset(x).lemmas()] for x in s]
        if len(s) == 0 :
            s = [[y.name() for y in x.lemmas()] for x in str]
        flt = [y for x in s for y in x]
        return flt
        
    #Antonyms - negation 
    def antonyms(self, string):
        str = wn.synsets(string)
        s = [x.name() for x in str]
        s = [x for x in s if string in x]
        s = [[y.antonyms()[0].name() for y in wn.synset(x).lemmas() if len(y.antonyms()) != 0 ] for x in s]
        #if len(s) == 0 :
        #    s = [[y.name() for y in x.lemmas()] for x in str]
        flt = [y for x in s for y in x]
        return flt
        
    def lexic(self, key = 'know'):
        #rt = self.filen[string]
        rt = self.dico[key]
        return rt['word']
        
    def find(self, string):
        #dic = dictionary()
        synonym = dictionary.synomyns(string)
        lexicon = dictionary.lexic(self, key = string)
        antonym = dictionary.antonyms(string)
        return synonym, lexicon, antonym 
        
    def syn(self, string):
        #dic = dictionary()
        synonym = dictionary.synomyns(string)
        return synonym
        
    def getFacit(self, question = 'facit1'):
        return self.facit[question]
        
    def getDoc(self, document = 'doc1'):
        return self.docs[document]
    #test only 
    def getRef(self):
        return self.reference
        
    def editFacit(self, change = '', question = 'facit11', index = '10',):
        filen = self.facit
        filen[question][index].append(change)
        
        #Save changes 
        fichier = open(self.fac, 'w')
        json.dump(filen, fichier, indent=2)
        fichier.close()
        return 0
    
    def editDocs(self, change = '',dox ='doc11'):
        filen = self.docs
        filen[dox].append(changes)
        
        #Save changes 
        fichier = open(self.doc, 'w')
        json.dump(filen, fichier, indent=2)
        fichier.close()
        
        return 0