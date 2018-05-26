import numpy as np
from numpy import *
from numpy import zeros
import arff
import json
from sklearn import tree
from data import data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mlxtend.plotting import plot_decision_regions
import itertools
from sklearn.model_selection import train_test_split
from Read_tsv_fredrik import read_data
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

question = data()

def read_feat():
    d = 'FEATURES_ALL_FINISHED.json'
    #d = 'FEATURES_ALL_TOTALS.json'
    f = open(d)
    filen = json.load(f)
    f.close()
    
    return filen['data']
    
def list_of_list_to_int(s):
    return np.array([[float(y) for y in x] for x in s])
  
def list_to_int(s):
    return np.array([int(x) for x in s])
    
def to_encode(y):
    from sklearn.preprocessing import MultiLabelBinarizer
    y = [[x] for x in y]
    return MultiLabelBinarizer().fit_transform(y)
   
def label_to_binary(x):

    list = []
    for y in x:
        if y == 2 or y == 3 :
            list.append(1)
        else :
            list.append(0)
    
    return list 

def randomforest(X,y,X_train, X_test, y_train, y_test, classifier = 'binary') :

    from sklearn.ensemble import RandomForestClassifier
    if classifier == 'binary' :
        clf = RandomForestClassifier(n_estimators=60, random_state = 0)
        modelCV = RandomForestClassifier(n_estimators=60, random_state = 0)
    else :
        clf = RandomForestClassifier(n_estimators=100, random_state = 0)
        modelCV = RandomForestClassifier(n_estimators=100, random_state = 0)
    
    randforest = clf.fit(X_train, y_train)
    y_pred = randforest.predict(X_test)
    
    #accuracy 
    accuracy = randforest.score(X_test, y_test)
    print("Accuracy : ", accuracy)
    
    #plot confusion matrix
    plot_confusion_heatmap(X,y,randforest)
    
    # CROSS VALIDATION :
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    scoring = 'accuracy'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    # result mean 
    mean_accuracy = results.mean()
    print("Cross validation : ", mean_accuracy)
    
    # confusion matrix :
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix :")
    print(confusion_matrix)
    
    # recall, precision and F-score
    
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))
    
def naivebayes(X,y,X_train, X_test, y_train, y_test, classifier = 'binary') :
 
    from sklearn.metrics.classification import accuracy_score, log_loss
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.naive_bayes import GaussianNB
    from sklearn import metrics
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    
    naive_clf = make_pipeline(PCA(), GaussianNB())
    naive_clf.fit(X_train, y_train)
    y_pred = naive_clf.predict(X_test)
    
    #accuracy 
    print('\nAccuracy :')
    print('{:.2%}\n'.format(metrics.accuracy_score(y_test, y_pred )))

    #plot confusion matrix
    plot_confusion_heatmap(X,y,naive_clf)
    
    # CROSS VALIDATION :
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    scoring = 'accuracy'
    modelCV = make_pipeline(PCA(), GaussianNB())
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    # result mean 
    mean_accuracy = results.mean()
    print("Cross validation : ", mean_accuracy)
    
    # confusion matrix :
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix :")
    print(confusion_matrix)
    
    # recall, precision and F-score
    
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))
    
def logistic(X,y,X_train, X_test, y_train, y_test, classifier = 'binary'):
    # train model for logistic regression 
    # and use the test to predict and 
    # calculate the accuracy of the obatined model
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    from sklearn import model_selection
    from sklearn.model_selection import cross_val_score
    #X, y = access_common_data()
    #logreg = LogisticRegression(C = 380, multi_class = 'multinomial', solver = 'newton-cg') 
    #logreg = LogisticRegression(C = 10, multi_class = 'multinomial', solver = 'newton-cg') 
    if classifier == 'binary' :
        logreg = LogisticRegression(C = 390) 
        modelCV = LogisticRegression(C = 390) 
    else :
        logreg = LogisticRegression(C = 240, multi_class = 'multinomial', solver = 'newton-cg') 
        modelCV = LogisticRegression(C = 240, multi_class = 'multinomial', solver = 'newton-cg') 
    #For bag of words model
    # if classifier == 'binary' :
    #     logreg = LogisticRegression(C = 10) 
    #     modelCV = LogisticRegression(C = 10) 
    # else :
    #     logreg = LogisticRegression(C = 10, multi_class = 'multinomial', solver = 'newton-cg') 
    #     modelCV = LogisticRegression(C = 10, multi_class = 'multinomial', solver = 'newton-cg') 
        
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    #accuracy 
    accuracy = logreg.score(X_test, y_test)
    print("Accuracy : ", accuracy)
    #plot confusion matrix
    plot_confusion_heatmap(X,y,logreg,classifier = 'multiclass')
    
    # CROSS VALIDATION :
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    scoring = 'accuracy'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    # result mean 
    mean_accuracy = results.mean()
    print("Cross validation : ", mean_accuracy)
    
    # confusion matrix :
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix :")
    print(confusion_matrix)
    
    # recall, precision and F-score
    
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))

    
def randomforest_train(X_train, y_train, cf = 10):

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=cf, random_state = 0)
    randforest = clf.fit(X_train, y_train)

    return randforest

def randomforest_test(randforest, X_test, y_test):
    
    y_pred = randforest.predict(X_test)
    accuracy = randforest.score(X_test, y_test)    
    return accuracy

def randomforest_test_cross(X_train, y_train, cf = 10):

    from sklearn import model_selection
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier

    # CROSS VALIDATION :
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    scoring = 'accuracy'
    modelCV = RandomForestClassifier(n_estimators=cf, random_state = 0)
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    # result mean 
    mean_accuracy = results.mean()
    
    return mean_accuracy    

def log_train(X_train, y_train, cf = 1):

    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    logreg = LogisticRegression(C = cf, multi_class = 'multinomial', solver = 'newton-cg') 
    logreg.fit(X_train, y_train)
    return logreg
    
def log_test(logreg, X_test, y_test):
    
    y_pred = logreg.predict(X_test)
    accuracy = logreg.score(X_test, y_test)    
    return accuracy

def log_test_cross(X_train, y_train, cf = 1, classifier = 'binary'):

    # Cross evaluation
    from sklearn.linear_model import LogisticRegression
    from sklearn import model_selection
    from sklearn.model_selection import cross_val_score
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    if classifier == 'binary' :
        modelCV = LogisticRegression(C = cf) 
    else :
        modelCV = LogisticRegression(C = cf, multi_class = 'multinomial', solver = 'newton-cg') 
    scoring = 'accuracy'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    mean_accuracy = results.mean()
    
    return mean_accuracy

def naive_test_cross(X_train, y_train, cf = 1, classifier = 'binary'):

    kfold = model_selection.KFold(n_splits=10, random_state=7)
    modelCV = make_pipeline(PCA(), GaussianNB())
    scoring = 'accuracy'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    mean_accuracy = results.mean()
    
    return mean_accuracy

def grid_search(X,y):
    
    # naive grid search implementation
    from sklearn.svm import SVC
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)    
    X_T, X_val, y_T, y_val = train_test_split(X_test,y_test, test_size=0.33, random_state=0)

    best_score = 0
    param = [x*10 for x in range(1,41)]
    param_naive = [ x  + 1 for x in range(1,18)]
    for C in param:
        # for each combination of parameters, train an SVC
        # clf = log_train(X_train, y_train, cf = C)
        # evaluate the SVC on the test set
        # score = log_test(clf, X_val, y_val)
        # if we got a better score, store the score and parameters
        # use cross validation
        # test logistic regression
        score = log_test_cross(X_val, y_val, cf = C)
        # test randmforest
        #score = randomforest_test_cross(X_val, y_val, cf = C)
        # test naive bayes
        #score = naive_test_cross(X_val, y_val, cf = C)
        print(score)
        if score > best_score:
            best_score = score
            best_parameters = {'C': C}
    print("Best score: {:.2f}".format(best_score))
    print("Best parameters: {}".format(best_parameters))


def feat_select_chi(X,y, k = 14):
    
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    
    X_new = SelectKBest(chi2, k).fit_transform(X, y)
    #X_new = SelectKBest(chi2, k = 16).fit_transform(X, y)
    return X_new, y
 
def feat_s3(x_train,y_train):
    from sklearn.svm import SVC
    from sklearn.datasets import load_digits
    from sklearn.feature_selection import RFECV
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Create the RFE object and rank each pixel
    # clf = RandomForestClassifier(n_estimators=100, random_state = 0)
    # feature selectin RandomForestClassifier para for multiclass
    #clf_rf_4 = RandomForestClassifier(n_estimators=100, random_state = 0)
    # feature selectin  RandomForestClassifier para for  binary
    #clf_rf_4 = RandomForestClassifier(n_estimators=60, random_state = 0)
    # clf_rf_4 = make_pipeline(PCA(n_components=18), GaussianNB())
    #clf_rf_4 = LogisticRegression(C = 390)
    clf_rf_4 =LogisticRegression(C = 10, multi_class = 'multinomial', solver = 'newton-cg') 
    rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
    rfecv = rfecv.fit(x_train, y_train)
        
    import matplotlib.pyplot as plt
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score of number of selected features")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

def plot_confusion_heatmap(X,y,clf_rf, classifier = 'binary'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score,confusion_matrix
    from sklearn.metrics import accuracy_score
    
    # split data train 70 % and test 30 %
    #x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)    
    X_test, _, y_test, _ = train_test_split(X_test,y_test, test_size=0.33, random_state=0)
    
    #random forest classifier with n_estimators=10 (default)
    # clf_rf = LogisticRegression()      
    # clr_rf = clf_rf.fit(x_train,y_train)
    
    ac = accuracy_score(y_test,clf_rf.predict(X_test))
    print('Accuracy is: ',ac)
    cm = confusion_matrix(y_test,clf_rf.predict(X_test))
    if classifier == 'binary' :
        class_names = np.array(['0','1'])
    else :
        class_names = np.array(['0','1','2','3'])
    plot_confusion_matrix(cm, classes=class_names, normalize=True)
    
    print(cm)
    # sns.heatmap(cm,annot=True,fmt="d")
    plt.show()
    
def corr_map(X_train):
    #correlation map
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    import pandas as pd
    # feats = ['cosine','nKeywords','lsa','partwords', 'lang_sim','lda','align','corpus_sim','lda_extract', 'cos_score','bingo_score','jaccard','dice sim', 'keywords_norm','LSI_query','bleuscore']
    # index = [x for x in range(16)]
    feats = ['nKeywords','lsa','partwords', 'lang_sim','lda','align','corpus_sim','lda_extract', 'cos_score','bingo_score','jaccard','dice sim', 'keywords_norm','LSI_query','bleuscore','Key','Ngram','Holistic']
    index = [x for x in range(18)]
    
    df = pd.DataFrame({'cosine': X_train[:,0]})
    for x in index :
        df[feats[x]] = pd.DataFrame({feats[x]: X_train[:,x]})
    
    f,ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(df.corr(), annot=True, linewidths=.9, fmt= '.1f',ax=ax)
    plt.show()

    
def plot_actual_predict(Y1, Y2, X = [0,1,2,3]):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.plot(X, Y1)
    plt.plot(X, Y2)
    plt.show()
    
def roc_curv(X, y, model = 'logistic'):
    
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle
    from sklearn.linear_model import LogisticRegression
    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier
    from scipy import interp
    
    y_true = list(y)
    # Binarize the output
    y = label_binarize(y, classes=[0, 1, 2, 3])
    print("y binary = ", y[0:10])
    n_classes = y.shape[1]
    
    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    
    # Learn to predict each class against the other
    if model == 'logistic' :
        # shuffle and split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=.5, random_state=0)
        logreg = LogisticRegression(C = 240, multi_class = 'multinomial', solver = 'newton-cg') 
    
        y_score = logreg.fit(X_train, y_train)
        y_score = y_score.decision_function(X_test)
        
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_test = label_binarize(y_test, classes=[0, 1, 2, 3])
        y_train = label_binarize(y_train, classes=[0, 1, 2, 3])
    elif model == 'randomforest' :
        # shuffle and split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=.5, random_state=0)

        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, random_state = 0)
        y_score = clf.fit(X_train, y_train).predict_proba(X_test)
        y_score = np.array(y_score)
        
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_test = label_binarize(y_test, classes=[0, 1, 2, 3])
        y_train = label_binarize(y_train, classes=[0, 1, 2, 3])
    elif model == 'bayes' :
        # shuffle and split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=.5, random_state=0)        
        naive_clf = make_pipeline(PCA(), GaussianNB())
        y_score = naive_clf.fit(X_train, y_train).predict_proba(X_test)

        y_score = np.array(y_score)
        
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_test = label_binarize(y_test, classes=[0, 1, 2, 3])
        y_train = label_binarize(y_train, classes=[0, 1, 2, 3])
        
        print()
    else :
        return 0
        
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    ## multi class
    
    lw = 2
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
    
    colors = cycle(['chartreuse', 'red', 'blue','indigo'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve multiclass logistic regression c = 240')
    plt.legend(loc="lower right")
    plt.show()

def roc_curv_binary(X, y, model = 'randomforest' ):
    
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle
    from sklearn.linear_model import LogisticRegression
    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier
    from scipy import interp
    
    y_true = list(y)
    # Binarize the output
    y = label_binarize(y, classes=[1, 2])
    n_classes = y.shape[1]
    print("nb_classes ",n_classes )
    
    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    
    # Learn to predict each class against the other
    if model == 'logistic' :
        # shuffle and split training and test sets
        #n_classes = 2
        X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=.5, random_state=0)
        logreg = LogisticRegression(C = 390) 
    
        y_score = logreg.fit(X_train, y_train)
        y_score = y_score.predict_proba(X_test)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_test = label_binarize(y_test, classes=[0, 1]).tolist()
        
        lista = []
        for y in y_test :
            if y == [0]:
                lista.append([1,0])
            if y == [1]:
                lista.append([0,1])
        y_test = np.array(lista)
        print(y_test[0:10])
        
        y_train = label_binarize(y_train, classes=[0, 1])
        print("np shape y_score = ", np.shape(y_score))
        print("np shape y_test = ", np.shape(y_test))
        
    elif model == 'randomforest' :
        #n_classes = 2
        X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=.5, random_state=0)

        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, random_state = 0)
        y_score = clf.fit(X_train, y_train).predict_proba(X_test)
        y_score = np.array(y_score)
        
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_test = label_binarize(y_test, classes=[0, 1]).tolist()
        
        lista = []
        for y in y_test :
            if y == [0]:
                lista.append([1,0])
            if y == [1]:
                lista.append([0,1])
        y_test = np.array(lista)
        print(y_test[0:10])
        y_train = label_binarize(y_train, classes=[0, 1])
        print("np shape y_score = ", np.shape(y_score))
        print("np shape y_test = ", np.shape(y_test))
    elif model == 'bayes' :

        # shuffle and split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=.5, random_state=0)        
        naive_clf = make_pipeline(PCA(), GaussianNB())
        y_score = naive_clf.fit(X_train, y_train).predict_proba(X_test)

        y_score = np.array(y_score)
        
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_test = label_binarize(y_test, classes=[0, 1]).tolist()
        lista = []
        for y in y_test :
            if y == [0]:
                lista.append([1,0])
            if y == [1]:
                lista.append([0,1])
        y_test = np.array(lista)
        print(y_test[0:10])
        y_train = label_binarize(y_train, classes=[0, 1])
    else :
        return 0
        
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        # fpr[i], tpr[i], _ = roc_curve(y_test[:], y_score[:])
        # roc_auc[i] = auc(fpr[i], tpr[i])
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    lw = 2
    # Plot all ROC curves
    plt.figure()
    colors = cycle(['blue', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Binary case random forest ')
    plt.legend(loc="lower right")
    plt.show()

def train_test_error(X_train, X_test,y_train, y_test):
    
    import numpy as np
    from sklearn import linear_model
    
    # #############################################################################
    # Generate sample data
    n_samples_train, n_samples_test, n_features = 75, 150, 500
    np.random.seed(0)
    coef = np.random.randn(n_features)
    coef[50:] = 0.0  # only the top 10 features are impacting the model
    X = np.random.randn(n_samples_train + n_samples_test, n_features)
    y = np.dot(X, coef)
    
    # #############################################################################
    # Compute train and test errors
    alphas = np.logspace(-5, 1, 60)
    enet = linear_model.ElasticNet(l1_ratio=0.7)
    train_errors = list()
    test_errors = list()
    for alpha in alphas:
        enet.set_params(alpha=alpha)
        enet.fit(X_train, y_train)
        train_errors.append(enet.score(X_train, y_train))
        test_errors.append(enet.score(X_test, y_test))
    
    i_alpha_optim = np.argmax(test_errors)
    alpha_optim = alphas[i_alpha_optim]
    print("Optimal regularization parameter : %s" % alpha_optim)
    
    # Estimate the coef_ on full data with optimal regularization parameter
    enet.set_params(alpha=alpha_optim)
    coef_ = enet.fit(X, y).coef_
    
    # #############################################################################
    # Plot results functions
    
    import matplotlib.pyplot as plt
    plt.subplot(2, 1, 1)
    plt.semilogx(alphas, train_errors, label='Train')
    plt.semilogx(alphas, test_errors, label='Test')
    plt.vlines(alpha_optim, plt.ylim()[0], np.max(test_errors), color='k',
            linewidth=3, label='Optimum on test')
    plt.legend(loc='lower left')
    plt.ylim([0, 1.2])
    plt.xlabel('Regularization parameter')
    plt.ylabel('Performance')
    
    # Show estimated coef_ vs true coef
    plt.subplot(2, 1, 2)
    plt.plot(coef, label='True coef')
    plt.plot(coef_, label='Estimated coef')
    plt.legend()
    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.26)
    plt.show()
    
def access_common_data(binary = False):
    fX, fy = read_data()
    data = np.array(read_feat())
    row, col = np.shape(data)
    y = list_to_int(data[:,-1])
    # Change labels from multiclass to binary two classes
    if binary :
        y = label_to_binary(y)
    X = list_of_list_to_int(data[:,0:col-1])
    X = np.array([x + y for (x,y) in zip(X.tolist(),fX)])
    #X,y = feat_select_chi(X,y, k = 15) 

    return X, y
    
def access_bagword_model(binary = False):

    d = 'bag_of_words_model_data.json'
    f = open(d)
    filen = json.load(f)
    f.close()
    X = filen['data']
    y = filen['labels']
    if binary :
        y = label_to_binary(y)
    return X,y
    
def compute_access_common_features(f = 'FEATURES_test.json'):
    """
    This function take too long timme 
    to comple. Used the precomputed set
    of features for test. Use the function 
    access_common_data() were the features
    are already computed and X and y are 
    ready to be used in any classifiers 
    """
    # from ml_model_features_computation import *
    from ml_model_features_computation import get_features
    X, y = get_features(f)
    
def tests(x = 1, feat = True):
    X, y = access_common_data()
    if feat :
        X,y = feat_select_chi(X,y, k = 14) 
    
    if x == 1 :
        X,y = X,y 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)    
        X_test,X_val, y_test, y_val = train_test_split(X_test,y_test, test_size=0.33, random_state=0)
        # X_test,X_val, y_test, y_val = train_test_split(X_test,y_test, test_size=0.33, random_state=0) 
        print()
        print("Train and test")
        print()
        logistic(X,y,X_train, X_test, y_train, y_test, classifier = 'multiclass')
        print()

    elif x == 2 :
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)    
        X_test,X_val, y_test, y_val = train_test_split(X_test,y_test, test_size=0.33, random_state=0)
        print()
        print("Train and test")
        print()
        randomforest(X,y,X_train, X_test, y_train, y_test)
        print()
        
    elif x == 3 :
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)    
        X_test,X_val, y_test, y_val = train_test_split(X_test,y_test, test_size=0.33, random_state=0)
        print()
        print("Train and test")
        print()
        naivebayes(X,y,X_train, X_test, y_train, y_test)
        print()

    elif x == 4 :
        grid_search(X,y)
    elif x == 5 :
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 
        train_test_error(X_train, X_test,y_train, y_test)
    elif x == 6 :
        corr_map(X_train)
    elif x == 7 :
        feat_s3(X_train,y_train)
    elif x == 8 :
        roc_curv(X,y)
        # roc_curv_binary(X,y)
    else:
        print("Hoj Hoj ! Bye Bye !")
        

# tests()




