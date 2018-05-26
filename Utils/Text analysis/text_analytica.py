import matplotlib.pyplot as plt
import numpy as np
from text_processing import *
from nltk import *
import numpy as np
from text_processing import *
from nltk.corpus import wordnet as wn
import json 
from data import data 

question = data()

def cum_plot_frequents_words(text, n = 50):
    """
    this takes as input a list of text 
    can be changed to plain text if wanted
    just uncomment add function
    
    """
    text = keep_text_hard(add(all_text))
    text = remove_stops(tokenizer(text))
    tokens = nltk.word_tokenize(text)
    clean = nltk.Text(tokens)
    fdist = FreqDist(clean)
    fdist.plot(n, cumulative=True)
   
    
def words_dispersion_plot(text):
    """
    this takes as input a list of text 
    can be changed to plain text if wanted
    just uncomment add function
    
    """
    facit = "need know quantity vinegar used container type sort cup sort type material test size surface area material used long time sample rinse distilled water dry method use temperature experiment"
    text = keep_text_hard(add(all_text))
    text = remove_stops(tokenizer(text))
    tokens = nltk.word_tokenize(text)
    clean = nltk.Text(tokens)
    
    clean.dispersion_plot(tokenizer(facit))
    
def unusual_words(text):
    
    text = keep_text_hard(add(all_text))
    text = remove_stops(tokenizer(text))
    tokens = nltk.word_tokenize(text)
    clean = nltk.Text(tokens)
    
    fdist1 = FreqDist(clean)
    return fdist1.hapaxes()

def text_collocation(text):
    """
    this takes as input a list of text 
    can be changed to plain text if wanted
    just uncomment add function
    
    """
    facit = "need know quantity vinegar used container type sort cup sort type material test size surface area material used long time sample rinse distilled water dry method use temperature experiment"
    text = keep_text_hard(add(all_text))
    text = remove_stops(tokenizer(text))
    tokens = nltk.word_tokenize(text)
    clean = nltk.Text(tokens)
    
    return clean.collocations()
    
def word_cloud(text):
    """
    this takes as input a list of text 
    can be changed to plain text if wanted
    just uncomment add function
    
    """
    #  pip install git+git://github.com/amueller/word_cloud.git
    #  https://github.com/amueller/word_cloud
    from wordcloud import WordCloud
    text = keep_text_hard(add(all_text))
    text = remove_stops(tokenizer(text))
    wordcloud = WordCloud(width = 1000, height = 500).generate(text)
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    
answers = question.get_question(0)
text = [ans['text'] for ans in answers]
text = ' '.join(text)
# print(text)

all_text = question.get_text_by_score(quest=0,score=0)
print(text_collocation(all_text))
print()
print(text_collocation(text))
# words_dispersion_plot(text)
#cum_plot_frequents_words(text)
#word_cloud(all_text)
