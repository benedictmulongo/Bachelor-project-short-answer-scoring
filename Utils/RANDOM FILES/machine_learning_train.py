import numpy as np
from numpy import *
from numpy import zeros
from features import feature_extraction
import arff
import json
from sklearn import tree
from data import data

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mlxtend.plotting import plot_decision_regions
import itertools
 
question = data()

def read_feat():
    
    d = 'FEATURES.json'
    f = open(d)
    filen = json.load(f)
    f.close()
    
    return filen['data']
    
def list_of_list_to_int(s):
    return np.array([[float(y) for y in x] for x in s])
  
def list_to_int(s):
    return np.array([int(x) for x in s])
    
data = np.array(read_feat()[0:10])
row, col = np.shape(data)
y = list_to_int(data[:,-1])
# X = list_of_list_to_int(data[:,0:col-1])
X = list_of_list_to_int(data[:,0:2])
print("Labels = ", y)
print()
print("Pattern = ", X)

# Parameters
n_classes = 4
plot_colors = "ryb"
plot_step = 0.02

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

plt.xlabel('Features')
plt.ylabel('Scores')

# Plot the training points
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label='scores',
                cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

# plt.suptitle("Decision surface of a decision tree using paired features")
# plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
plt.show()