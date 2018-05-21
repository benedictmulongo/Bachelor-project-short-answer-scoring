import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob as tb
from dictionary import dictionary 
from nltk.corpus import wordnet as wn
import json

from data import data 

dico = dictionary()
question = data()

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
    
# print(sentences_finder(text1))
def flatten(s):
    flt = [y for x in s for y in x]
    return flt
def filter(s):
    #re.split(';|,|\*|\n|\.|\/|',s)
    #k = [flatten([ y.split('.') for y in x ]) for x in s ]
    #added more split with regex
    k = [flatten([ re.split(';|,|\*|\n|\.|\/|',y) for y in x ]) for x in s ]
    return k
    
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
    
def cache(key = 'container', index = '1'):
    cacha = 'C:/Users/ben/Desktop/BachelorProject/cacher.json'
    f1 = open(cacha)
    cach = json.load(f1)
    f1.close()
    nltk = cach[index]['nltk'][key]
    dico = cach[index]['dico'][key]
    antonym = cach[index]['antonym'][key]
    
    return nltk, dico, antonym
    
def compute_score(word1,word2):
    score = 0
    #direct matching betwen words
    if str(word1) in str(word2) or str(word2) in str(word1) :
        #print(word1," ",word2 ," ",abs(len(word1) - len(word2)))
        if abs(len(word1) - len(word2)) <= 3:
            score = 1
    # *****************************************************************************
    # Add edit distance 
    # Delete digit 
    # delete one/two letter words 
    # *****************************************************************************
    #NLTK similirarity 
    nltk,dico,anto = cache(key = word2)
    in_nltk = word1 in nltk 
    in_dico = word1 in dico 
    in_anto = word1 in anto 
    score = in_nltk + in_dico + score 
    score = score/3
    if in_anto:
        score = -1
    return score
    
def lower_case(text):
    return text.lower()
def tokenizer(text):
    return word_tokenize(text)
def remove_stops(text):
    #added would also ?
    #stop = ['.',')','(','/','=','*','¨','%','€','|','&',',','also','would','?']
    stop = ['.',')','(','/','=','*','¨','%','€','|','&',',','also','would','?','.^']
    result = ""
    for x in text:
        if (x in stopwords.words('english')) != True and (x in stop) != True and len(x) > 1:
        #if (x in stopwords.words('english')) != True and (x in stop) != True:
            result = result + " " + x
    return result


def align(txt1, txt2, tokenize = True):
    
    text1 = txt1
    text2 = txt2
    txt1 = txt1
    txt2 = txt2
    textx = txt1
    texty = txt2
    
    if tokenize :
        txt1 = txt1.lower()
        txt2 = txt2.lower()
        
        text1 = tokenizer(txt1)
        text1 = tokenizer(remove_stops(text1))
        
        text2 = tokenizer(txt2)
        text2 = tokenizer(remove_stops(text2))
    
    
        textx = tokenizer(txt1)
        textx = tokenizer(remove_stops(textx))
        
        texty = tokenizer(txt2)
        texty = tokenizer(remove_stops(texty))
    
    score = 0
    par = []
    somme = []
    tot = 0
    for x in text1:
        for y in text2:
            score = compute_score(x,y)
            if score > 0.3:
                par.append((x,y,score))
                somme.append(score)
                tot = tot + score 
                text2.remove(y)
                break   

    total = sum(somme)

    result1 = (2*tot)/(len(textx) + len(texty))
    resultref1 = tot/(len(textx))
    resultref2 = len(somme)/(len(textx))
    allt = (result1 + resultref1 + resultref2)/3

    return allt
    
def clean_paragraph(text1, found_sent = True):
    X = text1
    if found_sent:
        X = sentences_finder(text1)
    x_lower = [x.lower() for x in X]
    x_tokens = [tokenizer(x) for x in x_lower]
    x_clean = [tokenizer(remove_stops(x)) for x in x_tokens]
    return x_clean
   
def duplist(one):
    vecteur = []
    for x in one:
        if x not in vecteur :
            vecteur.append(x)
    return len(vecteur)
    
def compute_all(text1, ref = ''):
    
    dico = dictionary()
    answers = clean_paragraph(text1)
    answers = duplicat(answers)
    #filt = filter(answers)
    answers = filter(answers)
    ref = clean_paragraph(dico.getDoc(), found_sent=False)
    print(" Answer : ", answers)
    print(" Ref : ", ref)
    #print("********************************1***************************************")
    list = []
    one = []
    for x in answers:
        for y in ref:
            score = align(x,y,tokenize = False)

            if score > 0.3:
                list.append((x,y,score))
                one.append(x)
            ref = clean_paragraph(dico.getDoc(), found_sent=False)

    #return list, len(list)
    return list, duplist(one)
    
    
# t = 'In order to replicate this experiment ,I would need to know how much vinegar to place in each container.Also I would need to know of each sample was, are solid piece or,n  small or pieces. The greas would need to identify's
# a, b = compute_all(t)
# print("Text ", 1 , " Score ", 3)
# # print("Result list = ", a)
# print("Result score? = ", b)

# def indexer 
    

all_text = question.get_text_by_score(quest=0,score=1)
count = 0
for tx in all_text :
    print("******************************** A ***************************************")
    count = count + 1
    a, b = compute_all(tx)
    print("Text ", count , " Score ", 1)
    #print("List one = ", a)
    print("Result score = ", b)
    print("******************************** B ***************************************")



#print(question.get_text_by_score(quest=0,score=3)[0:6])


# Text  2  Score  1
# List =  [(['amant', 'vinegar', 'poured', 'container'], ['need', 'know', 'much', 'quantity'], 0.5277777777777778), (['amant', 'vinegar', 'poured', 'container'], ['need', 'know', 'type', 'sort'], 0.5277777777777778), (['two'], ['need', 'know', 'much', 'quantity', 'vinegar', 'container'], 0.4761904761904762), (['two'], ['need', 'know', 'type', 'sort', 'vinegar', 'container'], 0.4761904761904762), (['two'], ['need', 'know', 'size', 'surface', 'area', 'material', 'materials'], 0.47222222222222215)]
# Result score =  5