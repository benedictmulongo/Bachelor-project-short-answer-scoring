import ngram
import nltk
from nltk.corpus import stopwords



facit_file = 'doc1_light.txt'
with open(facit_file, 'r') as file:
    facit = file.read().splitlines()

punctuation = ['.',')','(','/','=','*','¨','%','€','|','&',',']
stopWords = set(stopwords.words('english'))

def clean(data):
    data = [i for i in data if i not in stopWords]
    data = [i for i in data if i not in punctuation]
    return data

def bingo_give_score(answer):
    answer = clean(nltk.word_tokenize(answer))
    c = try_all(answer)
    c = sorted(c)[::-1]
    #return (c[0], c[1], c[2])
    return (c[0]+c[1]+c[2])

def try_all(answer):
    compound = list()
    for f in facit:
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
            
#X = clean(nltk.word_tokenize(X))
#Y = clean(nltk.word_tokenize(Y))


