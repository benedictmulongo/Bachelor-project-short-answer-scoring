import numpy as np
from sklearn.metrics import accuracy_score
import json 
from nltk.corpus import wordnet as wn
from data import data
import numpy as np
import json
from data import data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn import metrics
from rule_system import RuleSystem

question = data()
rulesys = RuleSystem()

def plot_confusion_heatmap_rule(y_pred, labels ,classifier = 'multiclass'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score,confusion_matrix
    from sklearn.metrics import accuracy_score

    ac = accuracy_score(labels,y_pred)
    print('Accuracy is: ',ac)
    cm = confusion_matrix(labels,y_pred)
    if classifier == 'binary' :
        class_names = np.array(['0','1'])
    else :
        class_names = np.array(['0','1','2','3'])
    plot_confusion_matrix(cm, classes=class_names, normalize=True)
    
    print(cm)
    plt.show()

def rule_classification_report(y_pred,labels):
    from sklearn.metrics import accuracy_score
    
    # Accuracy 
    accuracy = accuracy_score(labels,y_pred)
    print("Accuracy : ", accuracy)
    
    from sklearn.metrics import classification_report
    print(classification_report(labels,y_pred))

    
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
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
    

def perfomance(answers, category = 3):
    result = []
    for i, answer in enumerate(answers) :
        ret = rulesys.give_score(answer)
        result.append(ret)
        
    true_result = [category]*len(answers)
    accuracy = accuracy_score(result, true_result)
    return accuracy, result
    
def all_performance():
    ans0 = question.get_text_by_score(quest=0,score=0)
    ans1 = question.get_text_by_score(quest=0,score=1)
    ans2 = question.get_text_by_score(quest=0,score=2)
    ans3 = question.get_text_by_score(quest=0,score=3)
    answers = [ans0, ans1,ans2, ans3]
    
    length = len(ans0) + len(ans1) + len(ans2) + len(ans3)
    labels = [0]*len(ans0) + [1]*len(ans1) + [2]*len(ans2) + [3]*len(ans3)
    y_pred = []
    accs = []
    for cat, repons in enumerate(answers) :
        acc, pred = perfomance(repons, cat)
        y_pred = y_pred + pred
        print("Category = ", cat)
        print("Accuracy = ", acc)
        accs.append(acc)
    finalacc = sum(accs) / len(accs)
    print()
    print("Final Accuracy = ", finalacc)
    
    return y_pred, labels
  
def tests():
    y_pred, labels = all_performance()

    plot_confusion_heatmap_rule(y_pred, labels)
    print('**'*20)
    print("Classification report :\n")
    rule_classification_report(y_pred,labels)
    print("Hello ! can you hear me... ?")
    
#Test All Questions 
tests()

    




