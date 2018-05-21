import numpy as np
import ngram
from text_processing import *
from scipy.linalg import svd
import numpy as np
import json
from sklearn.model_selection import train_test_split

from data import data 
question = data()

def logistic(X_train, X_test, y_train, y_test):
    # Train model for logistic regression 
    # and use the test to predict and 
    # calculate the accuracy of the obatined model
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    logreg = LogisticRegression(C= 10, multi_class = 'multinomial', solver = 'newton-cg') 
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    
    # Accuracy 
    accuracy = logreg.score(X_test, y_test)
    print("Accuracy : ", accuracy)
    
    # Cross validation :
    from sklearn import model_selection
    from sklearn.model_selection import cross_val_score
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    modelCV = LogisticRegression(C= 10, multi_class = 'multinomial', solver = 'newton-cg') 
    scoring = 'accuracy'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    
    # Result mean 
    mean_accuracy = results.mean()
    print("Cross validation : ", mean_accuracy)
    
    # Confusion matrix :
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix :")
    print(confusion_matrix)
    
    # Recall, Precision and F-score
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))

def read_words():
    d = 'vocab.json'
    f = open(d)
    filen = json.load(f)
    f.close()
    return filen['vocab']

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    vocab_gram = ngram.NGram(vocabList)
    
    # for every word in the answer
    for word in inputSet:
        search = vocab_gram.search(word,threshold=0.5)
        if word in vocabList or len(search) > 0:
            if word in vocabList:
                returnVec[vocabList.index(word)] = 1
            else :
                s = search[0][0]
                returnVec[vocabList.index(s)] = returnVec[vocabList.index(s)] + 1
             #returnVec[vocabList.index(word)] = returnVec[vocabList.index(word)] + 1
        else : 
            c = 0
    return returnVec
    
def list_of_list_to_int(s):
    return np.array([[float(y) for y in x] for x in s])
  
def list_to_int(s):
    return np.array([int(x) for x in s])

def compute_features(answers = question.get_question(0)):
    data = []
    list_ans = []
    for index, answer in enumerate(answers) :
        
        score = answer['score_1']
        text = answer['text']
        ans = remove_stops(tokenizer(text))
        ans = tokenizer(ans)
        
        wordvec = setOfWords2Vec(read_words(), ans)
        summa = sum(wordvec)
        
        
        array = wordvec + [summa]  + [score]
        data.append(array)
        list_ans.append(text)

    return np.array(data), data, list_ans
    
def evaluate_model():
    
    answers = question.get_question(0)
    data,data_n,_ = compute_features(answers)
    row, col = np.shape(data)
    y = list_to_int(data[:,-1])
    X = list_of_list_to_int(data[:,0:col-1])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    logistic(X_train, X_test, y_train, y_test)

# evaluate_model()

