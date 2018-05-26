import numpy as np
import json 

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
    
    
data = np.array(read_feat())
row, col = np.shape(data)

# Send to a classifier 
y = list_to_int(data[:,-1])
X = list_of_list_to_int(data[:,0:col-1])

text = "dfhsdjhjsdf"

text = "dfhfsjfs"
text = "dfhfsjfs"
text = "dfhfsjfs"