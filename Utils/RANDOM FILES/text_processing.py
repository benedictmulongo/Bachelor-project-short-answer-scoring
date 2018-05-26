import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob as tb
from nltk.corpus import wordnet as wn
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
import json
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import nltk
from nltk.tokenize import sent_tokenize
#pip install fuzzywuzzy

def sent_tokenizer(text):
    sent_tokenize_list = sent_tokenize(text)
    return sent_tokenize_list

def remove_punctuations(text):
    words = nltk.word_tokenize(text)
    punt_removed = [w for w in words if w.lower() not in string.punctuation]
    return " ".join(punt_removed)
   
def add(text):
    s = ''
    for x in text:
        s = s + ' ' + x.lower()
    return s
    
def sentences_finder(text):
    #sentence finder blob 
    blob = tb(text)
    # grammatical correction
    blob = blob.correct()
    #***********************
    sent = blob.sentences
    ret = [ ''.join(x).strip() for x in sent]
    return ret

def correct_singularise(text, sing = False):
    """This function correct a text string and singularise and return also 
    a text string with variable ret ."""
    txt = tb(text.lower()).correct()
    if sing:
        txt = txt.words.singularize()
    else :
        txt = txt.words
    ret = ' '.join(txt).strip()
    return ret
    
def remove_shit(text):
    """This function takes a text string as input and remove digits  
    and others not so useful characters and return a text string"""
    data = re.sub('[-|0-9|\,|;|,|\*|\n|\/|\^|\.^|)|(]','',text.lower())
    return data
    
def lower_case(text):
    return text.lower()
    
def tokenizer(text):
    return word_tokenize(text)

def keep_text_hard(text):
    """This function takes a text string as input and remove all others
    characters that not alphanumerics and return a clean text string"""
    filtered_tokens = []
    tokens = tokenizer(text.lower())
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text
    
def keep_text(text):
    """This function takes a text string as input and remove all others
    characters that not alphanumerics and return a clean text string"""
    filtered_tokens = []
    tokens = tokenizer(text.lower())
    for token in tokens:
        if re.search('[a-zA-Z|.|,]', token):
            filtered_tokens.append(token)
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text
    
def flatten(s):
    flt = [y for x in s for y in x]
    return flt
    
def filter(s):
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
    
def remove_stops(text):
    stop = ['.',')','(','/','=','*','¨','%','€','|','&',',','also','would','?','.^']
    result = ""
    for x in text:
        if (x in stopwords.words('english')) != True and (x in stop) != True and len(x) > 2:
            result = result + " " + x.strip()
    return result.strip()
    

def porterstem(text):
    stemmer = PorterStemmer()
    text = correct_singularise(text)
    text = tokenizer(text)
    text = [stemmer.stem(tx) for tx in text]
    ret = ' '.join(text)
    
    return ret

def lemmatizer(text):
    lem = WordNetLemmatizer()
    text = correct_singularise(text)
    text = tokenizer(text)
    text = [lem.lemmatize(tx) for tx in text]
    ret = ' '.join(text)
    
    return ret
    
def ratio_text(text1, text2):
    return fuzz.ratio(text1, text2)
    
def partratio_text(text1, text2):
    return fuzz.partial_ratio(text1, text2)
    
def clean(text):
    data = re.sub(r'[-./?!,":;()\']',' ', text) 
    data = tokenizer(data)
    ret = ""
    for x in data:
        tx = x.strip()
        ret = ret + tx + " "
    return ret

def process_query(Query, Choices, N = 'one'):
    ret = []
    if N == 'one':
        ret = process.extractOne(query, choices)
    elif N == 'many':
        ret = process.extract(query, choices)
    return ret
    
def bigram_colloaction(text, N =  10):
    
    tmp = text
    tmp = remove_shit(tmp)
    tmp = remove_stops(tokenizer(tmp))
    words = [w.lower() for w in tokenizer(tmp)]
    bcf = BigramCollocationFinder.from_words(words)
    coll = bcf.nbest(BigramAssocMeasures.likelihood_ratio, N)   
    toget = [' '.join(r) for r in coll]
    text = ' '.join(toget)
    return text
    
def words_diff(text, minus):
    text = tokenizer(text)
    minus = tokenizer(minus)
    text_vocab = set(w.lower() for w in text if w.isalpha())
    minus_vocab = set(w.lower() for w in minus if w.isalpha())
    #diff = text_vocab - minus_vocab
    diff = minus_vocab - text_vocab
    #print(diff)
    diff = minus_vocab - diff
    return sorted(diff), ' '.join(sorted(diff)).strip()
    
def words_unusual(text, minus):
    text = tokenizer(text)
    minus = tokenizer(minus)
    text_vocab = set(w.lower() for w in text if w.isalpha())
    minus_vocab = set(w.lower() for w in minus if w.isalpha())
    diff = text_vocab - minus_vocab
    #diff = minus_vocab - text_vocab
    return ' '.join(sorted(diff)).strip()   
#0.75
#print("Fussy match :  ", ratio_text("vinegar","vinnagar"))
# text = "Beforee I wantedd 60 more potatoees but it was already harder] , [ = . to find only 10"
#     
# print(keep_text(text))
# print(porterstem(text))
# print(lemmatizer(text))

# return nouns, verbs