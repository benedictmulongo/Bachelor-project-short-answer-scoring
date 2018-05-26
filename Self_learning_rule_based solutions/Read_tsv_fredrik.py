import pandas as pd
import csv
import json
import numpy as np

def get_data(question=1):
    data = list()
    #data_file = 'FEATVECTOR_FREDRIK.tsv'
    data_file = 'new_features_fredrik.tsv'
    with open(data_file) as file:
        reader = csv.reader(file, delimiter='\t')
        count = 0
        for row in reader:
            data.append(row[1:])
    return data
    
def list_of_list_to_int(s):
    return np.array([[float(y) for y in x] for x in s])
  
def list_to_int(s):
    return np.array([int(x) for x in s])
    
def read_data():
    
    data = np.array(get_data()[1:])
    row, col = np.shape(data)
    y = list_to_int(data[:,-1])
    X = list_of_list_to_int(data[:,0:col-1])
    
    return X.tolist(), y.tolist()

# print(read_data())