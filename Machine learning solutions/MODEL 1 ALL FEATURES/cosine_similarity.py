from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import scipy.sparse as sp 
import numpy as np
from text_processing import *

def clean_paragraph(text1):
    
    X = text1
    x_lower = [x.lower() for x in X]
    x_clean = [remove_stops(tokenizer(x)) for x in x_lower]
    return x_clean
    
def compute_cosine_similarity(doc_features, corpus_features,top_n=7):
    
    # get document vectors
    doc_features = doc_features.toarray()[0]
    corpus_features = corpus_features.toarray()
    # compute similarities
    similarity = np.dot(doc_features,corpus_features.T)
    # get docs with highest similarity scores
    top_docs = similarity.argsort()[::-1][:top_n]
    top_docs_with_score = [(index, round(similarity[index], 3)) for index in top_docs]
    return top_docs_with_score


def compute_cosine(answer):
    
    toy_corpus = ["amount much quantity vinegar acid used each cup container",
        "type sort vinegar was used each cup container",
        "kind sort type material materials test",
        "size surface area material materials used",
        "long time minute minutes each sample was rinsed rinse distilled water",
        "dry drying method procedure approach",
        "size type container cup jar kettle use", 
        "temperature degree experiment" 
    ]
    
    norm_corpus = clean_paragraph(toy_corpus)
    min_df=0.0
    ngram_range=(1, 1)
    max_df=1.0
    tfidf_vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    tfidf_features =tfidf_vectorizer.fit_transform(norm_corpus).astype(float)
    
    query_docs_tfidf = tfidf_vectorizer.transform([answer])
    top_similar_docs = compute_cosine_similarity(query_docs_tfidf, tfidf_features, top_n=1)
    
    return top_similar_docs[0][1]
    
#print(compute_cosine("I need to know how much vinegar were used"))

# use the function compute(answer)
# where answer is the student input
# text

