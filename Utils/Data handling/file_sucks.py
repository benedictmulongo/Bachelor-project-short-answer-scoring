file = 'C:/Users/ben/Desktop/BachelorProject/paraphrase/ppdb-2.0-xxl-lexical'
fri = 'C:/Users/ben/Desktop/BachelorProject/paraphrase/ppdb-2.0-s-syntax'
fr = 'C:/Users/ben/Desktop/BachelorProject/paraphrase/ppdb-2.0-s-phrasal'

import json
from dictionary import dictionary 
facit = {"facit1":{"1":['need know quantity vinegar was used in each container'],"2":['need know sort amount vinegar used in each container'],"3":['need know what sort type materials sample test'],"4":['need know what size surface area material should used'],"5":['need know long time each sample was rinsed in distilled water'],"6":['need know what which drying method use'],"7":['need know what which size  sort container use']},"facit2":{"1":[],"2":[],"3":[],"4":[],"5":[],"6":[],"7":[],"8":[],"9":[]},"facit3":{"1":[],"2":[],"3":[],"4":[],"5":[],"6":[],"7":[],"8":[],"9":[]},"facit4":{"1":[],"2":[],"3":[],"4":[],"5":[],"6":[],"7":[],"8":[],"9":[]},"facit5":{"1":[],"2":[],"3":[],"4":[],"5":[],"6":[],"7":[],"8":[],"9":[]},"facit6":{"1":[],"2":[],"3":[],"4":[],"5":[],"6":[],"7":[],"8":[],"9":[],"10":[],"11":[]},"facit7":{"1":[],"2":[],"3":[],"4":[],"5":[],"6":[],"7":[]},"facit8":{"1":[],"2":[],"3":[],"4":[],"5":[],"6":[],"7":[]},"facit9":{"1":[],"2":[],"3":[],"4":[],"5":[],"6":[],"7":[]},"facit10":{"1":[],"2":[],"3":[],"4":[],"5":[],"6":[],"7":[],"8":[],"9":[],"10":[]}}

reference = 'You need to know how much vinegar was used in each container. You need to know what type of vinegar was used in each container. You need to know what materials to test. You need to know what size/surface area of materials should be used. You need to know how long each sample was rinsed in distilled water. You need to know what drying method to use. You need to know what size/type of container to use.'
doc1 = ['need know how much quantity vinegar was used in each container','need know what type sort vinegar was used in each container','need to know what sort type materials test','need know what size surface area material materials should used','know long time each sample was rinsed rinse distilled water','need to know what which drying method use','know which what size type container use']

# facit_keys = ['need', 'know', 'much', 'quantity', 'vinegar', 'used', 'container', 'need', 'know', 'type', 'sort', 'vinegar', 'used', 'container', 'need', 'know', 'sort', 'type', 'materials', 'test', 'need', 'know', 'size', 'surface', 'area', 'material', 'materials', 'used', 'know', 'long', 'time', 'sample', 'rinsed', 'rinse', 'distilled', 'water', 'need', 'know', 'drying', 'method', 'use', 'know', 'size', 'type', 'container', 'use']

doc = {"doc1":doc1,"doc2":[],"doc3":[],"doc4":[],"doc5":[],"doc6":[],"doc7":[],"doc8":[],"doc9":[],"doc10":[]}

cache = {"1":{"nltk":{}, "dico":{}, "antonym":{} },"2":{"nltk":{}, "dico":{}, "antonym":{} },"2":{"nltk":{}, "dico":{}, "antonym":{} },"3":{"nltk":{}, "dico":{}, "antonym":{} },"4":{"nltk":{}, "dico":{}, "antonym":{} },"5":{"nltk":{}, "dico":{}, "antonym":{} },"6":{"nltk":{}, "dico":{}, "antonym":{} },"7":{"nltk":{}, "dico":{}, "antonym":{} },"8":{"nltk":{}, "dico":{}, "antonym":{} },"9":{"nltk":{}, "dico":{}, "antonym":{}},"10":{"nltk":{}, "dico":{}, "antonym":{}}}


def init_dict(fil):
    dictionary = {}
    idem = {"word":[],"pos":'NN'}
    infile = open(fil, "r")
    for line in infile.readlines():

        sent = line.split("|||")
        pos = str(sent[0][1:len(sent[0])-2])
        a = str(sent[1]).strip()
        b = str(sent[2]).strip()
        #print("a = ", a, ", b = ", b)
        dictionary[a] = idem
        dictionary[b] = idem
        
    print("****************************************************************")
    #print(dictionary['modeling'])
    
    #dict, idem = init_dict(file)
    # f = open('dictionary.json', 'w')
    # json.dump(dict, f, indent=2)
    # f.close()
    f = open('syntax.json', 'w')
    json.dump(dictionary, f, indent=2)
    f.close()
    infile.close()
    return dictionary, idem
    
def update_dict(dictionary = 'C:/Users/ben/Desktop/BachelorProject/syntax.json'):
    
    f = open(dictionary)
    filen = json.load(f)
    f.close()
    
# >> import json
# >>> f = open('car.json')
# >>> car = json.load(f)
# >>> f.close()
# >>> car['mycar']['color'] = 'red'
# >>> f = open('car.json', 'w')
# >>> json.dump(car, f)
# >>> f.close()
    infile = open(file, "r")
    for line in infile.readlines():

        sent = line.split("|||")
        pos = str(sent[0][1:len(sent[0])-2])
        a = str(sent[1]).strip()
        b = str(sent[2]).strip()
        #print("a = ", a, ", b = ", b)
        filen[a]['word'].append(b)
        filen[a]['pos'] = pos
        filen[b]['word'].append(a)

    print("****************************************************************")
    
    #open a new file to save changes
    fichier = open('syntaxes.json', 'w')
    json.dump(filen, fichier, indent=2)
    fichier.close()
    print("Filen ord : ")
    print(filen['debated'])


    print("****************************************************************")
    return 0

def find(dir = 'C:/Users/ben/Desktop/BachelorProject/dico.json', key = 'wash'):
    f = open(dir)
    filen = json.load(f)
    f.close()
    print(filen[key])
    print('sample' in filen[key]['word'])
    
def write_facit():
    f = open('facit.json', 'w')
    json.dump(facit, f, indent=2)
    f.close()
    
def write_docs():
    f = open('docs.json', 'w')
    json.dump(doc, f, indent=2)
    f.close()

def write_cache():
    #nltk,dico,anto = dico.find(word2)
    print("*************************** BEGIN ********************************")
    facit_keys = ['need', 'know', 'much', 'quantity', 'vinegar', 'used', 'container', 'need', 'know', 'type', 'sort', 'vinegar', 'used', 'container', 'need', 'know', 'sort', 'type', 'materials', 'test', 'need', 'know', 'size', 'surface', 'area', 'material', 'materials', 'used', 'know', 'long', 'time', 'sample', 'rinsed', 'rinse', 'distilled', 'water', 'need', 'know', 'drying', 'method', 'use', 'know', 'size', 'type', 'container', 'use']
    from dictionary import dictionary 
    count = 0
    for y in facit_keys:
        dico = dictionary()
        #cache['1']['nltk'][y] = dico.synomyns(y)
        cache['1']['dico'][y] = dico.lexic(y)
        cache['1']['antonym'][y] = dico.antonyms(y).append('not ' + y)
        count = count + 1
        print("*************************** ",count ," ********************************")
    print("*************************** | END | ********************************")
    f = open('cache_dico.json', 'w')
    json.dump(cache, f, indent=2)
    f.close()

def find_all(key = 'container', index = '1'):
    cacha = 'C:/Users/ben/Desktop/BachelorProject/dico_cache.json'
    f1 = open(cacha)
    cach = json.load(f1)
    f1.close()
    nltk = cach[index]['nltk'][key]
    dico = cach[index]['dico'][key]
    antonym = cach[index]['antonym'][key]
    
    from dictionary import dictionary 
    dico = dictionary()
    a,b,c = dico.find(key)
    
    print(" Cache NLTK = ", nltk)
    print(" Cache Dico = ", dico)
    print(" Cache Anto = ", antonym)
    
    print(" Dictionary NLTK = ", a)
    print(" Dictionary Dico = ", b)
    print(" Dictionary Anto = ", c) 
    
#find_all()
#write_cache()
# init_dict(fri)
# update_dict()
#write_facit()
#write_docs()
# find(key='.')
#print("*****&&&&&&&&&&&&&&&&&&&&&&&&&*****")
#find(key='proteins')
#find(key='which')

# dico = dictionary()
# a,b,c = dico.find('living')
# print("NLTK = ", a)
# print("********************************************************************")
# print("Dico_para  = ", b)
# print("********************************************************************")
# print("Antonyms = ", c)
# print("********************************************************************")
# print("facti = ", dico.getFacit())
# print("********************************************************************")
# print("Docs = ", dico.getDoc())
  
# a = dico.find('know')
# print("NLTK = ", a) 
    
    