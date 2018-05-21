import csv
import os

def get_facit(question=1):
    facit_file = 'facit_' + str(question) + '.txt'
    with open(facit_file) as file:
        facit = file.read().splitlines()
    return facit

def get_ref(question=1):
    ref_file = 'ref_' + str(question) + '.txt'
    with open(ref_file) as file:
        references = file.read().splitlines()
    return references

def get_data(question=1):
    data = list()
    data_file = 'tb_data_' + str(question) + '.tsv'
    with open(data_file) as file:
        reader = csv.DictReader(file, dialect='excel-tab')
        for row in reader:
            data.append(row)
    return data

def get_ben_features():
    import json
    import numpy as np
    name = 'dataset/FEATURES_ALL_FINISHED.json'
    f = open(name)
    filen = json.load(f)
    f.close()

    def list_of_list_to_int(s):
        return np.array([[float(y) for y in x] for x in s])
  
    def list_to_int(s):
        return np.array([int(x) for x in s])
    
    data = np.array(filen['data'])
    row, col = np.shape(data)
    # Send to a classifier 
    scores = list_to_int(data[:,-1])
    features = list_of_list_to_int(data[:,0:col-1])
    
    return features.tolist(), scores.tolist()

    def mod_special():
        print('not done')

if __name__ == "__main__":
    get_facit()
    get_ref()
    get_data()
    print('OK')