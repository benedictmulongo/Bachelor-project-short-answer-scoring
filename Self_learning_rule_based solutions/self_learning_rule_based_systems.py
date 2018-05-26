import numpy as np
import matplotlib.pyplot as plt
import itertools
import json
from sklearn.model_selection import train_test_split
from data import data
from part3_view1 import compute_features
from ML_TRAIN_GATHER_STATS import access_common_data

question = data()
from rule_system import RuleSystem
rulesys = RuleSystem()

def list_of_list_to_int(s):
    return np.array([[float(y) for y in x] for x in s])
  
def list_to_int(s):
    return np.array([int(x) for x in s])
    
def divide_data(X,y, part = [0.3, 0.57, 0.33]):
    
    # X, label_X, y, label_y = train_test_split(X, y, test_size=0.3, random_state=0)
    # X, unlabel_X, y, unlabel_y = train_test_split(X, y, test_size=0.57, random_state=0)
    # X_test, val_X, y_test, val_y = train_test_split(X, y, test_size=0.33, random_state=0)
    
    X, label_X, y, label_y = train_test_split(X, y, test_size=part[0], random_state=0)
    X, unlabel_X, y, unlabel_y = train_test_split(X, y, test_size=part[1], random_state=0)
    X_test, val_X, y_test, val_y = train_test_split(X, y, test_size=part[2], random_state=0)
    return [label_X, label_y], [unlabel_X, unlabel_y], [val_X, val_y], [X_test, y_test]
    
def random_data():
    import random
    
    x = [ x for x in range(100)]
    label = []
    data = []
    for i in range(40):
        y = random.random()
        if y > 0.5 :
            label.append(1)
        else :
            label.append(0)
        dat = random.sample(x,  6)
        data.append(dat)
    return data, label

def log_train(X_train, y_train):

    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    logreg = LogisticRegression(C= 240, multi_class = 'multinomial', solver = 'newton-cg') 
    # logreg = LogisticRegression() 
    logreg.fit(X_train, y_train)
    return logreg
    
def log_test(logreg, X_test, y_test, report = False):
    
    y_pred = logreg.predict(X_test)
    accuracy = logreg.score(X_test, y_test)   
    from sklearn.metrics import classification_report

    if report :
        print(classification_report(y_test, y_pred)) 
        print("-------------------------")
        print()
        plot_confusion_heatmap(y_pred, y_test)
    return accuracy

def log_test_cross(logreg, X_train, y_train, X_test, y_test):

    # Cross evaluation
    from sklearn.linear_model import LogisticRegression
    from sklearn import model_selection
    from sklearn.model_selection import cross_val_score
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    # modelCV = LogisticRegression() 
    modelCV = LogisticRegression(C= 240, multi_class = 'multinomial', solver = 'newton-cg') 
    scoring = 'accuracy'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    mean_accuracy = results.mean()
    
    return mean_accuracy

def logistic(X_train, X_test, y_train, y_test):
    # train model for logistic regression 
    # and use the test to predict and 
    # calculate the accuracy of the obatined model
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    logreg = LogisticRegression(C= 240, multi_class = 'multinomial', solver = 'newton-cg') 
    #logreg = LogisticRegression() 
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    
    #accuracy 
    accuracy = logreg.score(X_test, y_test)
    
    from sklearn import model_selection
    from sklearn.model_selection import cross_val_score
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    modelCV = LogisticRegression() #380
    scoring = 'accuracy'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    # result mean 
    
    mean_accuracy = results.mean()
    return accuracy, mean_accuracy    
    
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

def ensemble_votes(test_X,test_y,ensemble ):
    from sklearn.metrics import accuracy_score
    y_pred = [] 
    for ex in test_X:
        votes = []
        for voter in ensemble :
            pred = voter.predict(np.array(ex).reshape(1,-1))
            votes.append(int(pred))
        #print(votes)
        # Return the majority element / most frequent score
        majority = np.bincount(votes).argmax()
        y_pred.append(majority)
    
    ac = accuracy_score(test_y,y_pred)
    print("Accuracy MAJORITY VOTES = ", ac)
    
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',
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
    
def plot_confusion_heatmap(y_pred, labels, classifier = 'multiclass'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score,confusion_matrix
    from sklearn.metrics import accuracy_score
    
    ac = accuracy_score(labels,y_pred)
    print('Accuracy is plot: ',ac)
    cm = confusion_matrix(labels,y_pred)
    if classifier == 'binary' :
        class_names = np.array(['0','1'])
    else :
        class_names = np.array(['0','1','2','3'])
    plot_confusion_matrix(cm, classes=class_names, normalize=True)
    
    print(cm)
    plt.show()
    plt.figure()

def self_learning(X,y, ans_text, Iteration = 5):
    # Divide the data in labeled set, unlabeled set, validation and 
    # test set 
    X1 = list(X)
    # 0.2 L 0.2 U 0.5 Test 0.1 Val
    partition1 = [0.2, 0.25, 0.16]
    # 0.1 L 0.2 U 0.6 Test 0.1 Val
    partition2 = [0.1, 0.22, 0.143]
    # 0.1 L 0.6 U 0.2 Test 0.1 Val
    partition3 = [0.1, 0.67,0.33]
    # 0.1 L 0.2 U 0.6 Test 0.1 Val
    partitionx = [0.1, 0.22, 0.14]
    # 0.1 0.1 0.7 0.1
    partition_final = [0.1,0.11,0.14]
    #labeled, unlabeled, val, test = divide_data(X,y, part = partition_final)
    partition_nein = [0.2,0.5,0.33]
    # labeled, unlabeled, val, test = divide_data(X,y, part = partition_nein)
    labeled, unlabeled, val, test = divide_data(X,y, part = partitionx)
    # Labeled set for initial training of the algorithm
    labeled_X = labeled[0]
    labeled_y = labeled[1]
    
    # Unlabeled for automatic discovery labels using
    # the labeled set 
    unlabeled_X = unlabeled[0]
    unlabeled_y = unlabeled[1]
    
    # Validation for benchmark purpose and termination
    # criterion implementation
    val_X = val[0]
    val_y = val[1]
    
    # Test for accuracy measure of the algorithm performance
    test_X = test[0]
    test_y = test[1]
    
    # Copy the unlabeled to be used in each 
    # iteration
    T_X = list(labeled_X)
    T_y = list(labeled_y)
    U = list(unlabeled_X)
    
    logreg = log_train(T_X, T_y)
    accuracy = log_test(logreg, test_X, test_y)
    cross_accuracy = log_test_cross(logreg,T_X, T_y,test_X, test_y)
    
    print("First accuracy = ", accuracy)
    print("Cross accuracy = ", cross_accuracy)
    iter = Iteration
    # Ensemble classifier 
    ensemble = []
    stop_halt = 0
    while len(U) > 0 and iter > 0 :
        #train a classifier on T
        logreg = log_train(T_X, T_y)
        # Save classifier for majority vote
        ensemble.append(logreg)
        # test on validation set 
        accuracy = log_test(logreg, val_X, val_y)
        cross_accuracy = log_test_cross(logreg,T_X, T_y,val_X, val_y)
        print("(Count down) Iteration NR = ", iter)
        for i, x in enumerate(U) :
            # Get index of unlabeled examples
            # better optimized
            index = X1.index(x)
            rule = rulesys.give_score(ans_text[index])
            y_pred = logreg.predict(np.array(x).reshape(1,-1))

            if rule == y_pred[0]:

                T_X.append(x)
                T_y.append(y_pred[0])
                # Delete the labeled example
                del U[i]
                
        # Decrease the iteration count
        iter = iter - 1
        
        accuracy = log_test(logreg, val_X, val_y)
        print("Val accuracy = ", accuracy)
        
        if stop_halt < accuracy:
            stop_halt = accuracy
        else :
            break
    
    ## Test majority votes 
    ensemble_votes(test_X,test_y,ensemble ) 
    
    print()
    #accuracy = log_test(logreg, test_X, test_y, True)
    accuracy = log_test(logreg, test_X, test_y)
    cross_accuracy = log_test_cross(logreg,T_X +test_X , T_y + test_y,test_X, test_y)
    
    print("self_learning accuracy = ", accuracy)
    print("self_learningCross accuracy = ", cross_accuracy)
    print("Length of self_learning = ", len(T_X))
    
    cross_accuracy = log_test_cross(logreg,T_X, T_y,test_X, test_y)
    print("Data only  self_learning all Cross accuracy = ", cross_accuracy)


    X_all = labeled_X + unlabeled_X
    y_all = labeled_y + unlabeled_y
    logreg = log_train(X_all, y_all)
    #accuracy = log_test(logreg, test_X, test_y, True)
    accuracy = log_test(logreg, test_X, test_y)
    cross_accuracy = log_test_cross(logreg,X_all + test_X, y_all + test_y,test_X, test_y)
    print("All accuracy = ", accuracy)
    print("All Cross accuracy = ", cross_accuracy)
    print("Length of self_learning = ", len(X_all))
    
    cross_accuracy = log_test_cross(logreg,X_all, y_all,test_X, test_y)
    print("Data only all Cross accuracy = ", cross_accuracy)

def bow_precomputed_evaluation() :
    data_np, data, ans = compute_features()
    X, y = access_bagword_model()   
    # self_learning(X.tolist(),y.tolist(), ans)
    self_learning(X,y, ans)
    
def bow_evaluation() :
        
    data_np, data, ans = compute_features()
    row, col = np.shape(data)
    y = list_to_int(data_np[:,-1])
    X = list_of_list_to_int(data_np[:,0:col-1])
    self_learning(X,y, ans)

def ml_feat_evaluation():
    
    # X, y = random_data()
    data_np, data, ans = compute_features()
    X,y = access_common_data()
    self_learning(X.tolist(),y.tolist(), ans)
    
def tests(x=3):
    if x == 1 :
        print()
        bow_precomputed_evaluation()
    elif x == 2 :
        print()
        bow_evaluation()
    else :
        print()
        ml_feat_evaluation()

tests()