import ngram
import nltk
from nltk.corpus import stopwords
import sys
sys.path.append('..')
import data_handler as dh

class Fgram:

    def __init__(self):
        self.facit = dh.get_facit(1)
        self.punctuation = ['.',')','(','/','=','*','¨','%','€','|','&',',']
        self.stopWords = set(stopwords.words('english'))

    def clean(self, data):
        data = [i for i in data if i not in self.stopWords]
        data = [i for i in data if i not in self.punctuation]
        return data

    def give_score(self, answer):
        answer = self.clean(nltk.word_tokenize(answer))
        c = self.try_all(answer)
        c = sorted(c)[::-1]
        #return (c[0], c[1], c[2])
        return (c[0]+c[1]+c[2])

    def try_all(self, answer):
        compound = list()
        for f in self.facit:
            f = nltk.word_tokenize(f)
            G = ngram.NGram(f)
            fas = list()
            for a in answer:
                fas.extend(G.search(a))
            s = 0
            for word, weight in fas:
                s = s + weight
            compound.append(s)
        return compound