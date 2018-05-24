import nltk
from nltk.corpus import wordnet as wn
import csv

class RuleSystem:

    #def __init__(self):

    def find_test_seq(self, answer):

        answer = answer.lower()
        answer = nltk.word_tokenize(answer)

        # Extend the words in the student answer with more lemmas.
        answer = [self.extend(w) for w in answer]
        # Combine 'list of list' to a 'list'
        answer = [item for sublist in answer for item in sublist]

        sequence = list()
        if 'vinegar' in answer:
            sequence += [0,1,2]
        if 'materials' in answer:
            sequence += [3,4]
        if 'material' in answer:
            sequence += [3,4]
        if 'container' in answer:
            sequence += [1,7,8]
        if 'drying' in answer:
            sequence += [6]
        if 'temperature' in answer:     # FIXME add temperature
            sequence += [10]
        if 'sample' in answer:
            sequence += [5,9]
        if 'rinsed' in answer:
            sequence += [5]

        sequence = list(set(sequence))
        sequence.sort()
        return sequence

    def extend(self, word):
        synset = wn.synsets(word)
        if synset:
            lemmas = synset[0].lemma_names()
            return lemmas
        else:
            return [word]

    def sentance_sequence(self, answer):
        sentances = nltk.sent_tokenize(answer)
        result = [(s, self.find_test_seq(s)) for s in sentances]
        return result


    def context(self, answer, index):
        answer = nltk.pos_tag(nltk.word_tokenize(answer), tagset='None')
        #print(answer)

        points = list()
        if index is 0:
            points += [nltk.pos_tag(nltk.word_tokenize('how much vinegar'), tagset='None')]
            points += [nltk.pos_tag(nltk.word_tokenize('amount of vinegar'), tagset='None')]
        if index is 1:
            points += [nltk.pos_tag(nltk.word_tokenize('used in each container'), tagset='None')]
            #FIXME this one should be removed
        if index is 2:
            points += [nltk.pos_tag(nltk.word_tokenize('type of vinegar'), tagset='None')]
            points += [nltk.pos_tag(nltk.word_tokenize('sort of vinegar'), tagset='None')]
            points += [nltk.pos_tag(nltk.word_tokenize('brand of vinegar'), tagset='None')]
        if index is 3:
            points += [nltk.pos_tag(nltk.word_tokenize('materials to test'), tagset='None')]
            points += [nltk.pos_tag(nltk.word_tokenize('material to test'), tagset='None')]
        if index is 4:
            points += [nltk.pos_tag(nltk.word_tokenize('surface area of materials'), tagset='None')]
            points += [nltk.pos_tag(nltk.word_tokenize('surface area of material'), tagset='None')]
            points += [nltk.pos_tag(nltk.word_tokenize('materials surface area'), tagset='None')]
            points += [nltk.pos_tag(nltk.word_tokenize('material surface area'), tagset='None')]
        if index is 5:
            points += [nltk.pos_tag(nltk.word_tokenize('how long each sample was rinsed'), tagset='None')]
            points += [nltk.pos_tag(nltk.word_tokenize('time to rinsed'), tagset='None')]
            points += [nltk.pos_tag(nltk.word_tokenize('how long rinsed'), tagset='None')]
        if index is 6:
            points += [nltk.pos_tag(nltk.word_tokenize('drying method to use'), tagset='None')]
            points += [nltk.pos_tag(nltk.word_tokenize('method of drying'), tagset='None')]
        if index is 7:
            points += [nltk.pos_tag(nltk.word_tokenize('size of container'), tagset='None')]
            points += [nltk.pos_tag(nltk.word_tokenize('container size'), tagset='None')]
            points += [nltk.pos_tag(nltk.word_tokenize('proportion of container'), tagset='None')]
        if index is 8:
            points += [nltk.pos_tag(nltk.word_tokenize('type of container'), tagset='None')]
            points += [nltk.pos_tag(nltk.word_tokenize('sort of container'), tagset='None')]
            points += [nltk.pos_tag(nltk.word_tokenize('brand of container'), tagset='None')]
        if index is 9:
            points += [nltk.pos_tag(nltk.word_tokenize('surface area of sample'), tagset='None')]
            points += [nltk.pos_tag(nltk.word_tokenize('sample surface area'), tagset='None')]
        if index is 10:
            points += [nltk.pos_tag(nltk.word_tokenize('temperature of the room'), tagset='None')]
            points += [nltk.pos_tag(nltk.word_tokenize('the rooms temperature'), tagset='None')]

        have = False
        for statment in points:  # try each statement
            n_pos = len(statment)
            for i in range(len(answer)-n_pos):
                if self.find_similarity(answer[i:i+n_pos], statment, n_pos):
                    #print('lord')
                    have = True
                    break
            if have: 
                break
        return have
    
    def have_in_it(self, answer, statement, n):
        answer = [a[1] for a in answer]
        statment = [s[1] for s in statement]
        have = True
        for s in statment:
            if s not in answer:
                have = False
        return have
    
    def find_similarity(self, answer, statment, n):
        result = True
        #print(answer)
        #print(statment)
        for i in range(n):
            #print('answer[' + str(i) +'][1]: ' + str(answer[i][1]))
            #print('statment[' + str(i) +'][1]: ' + str(statment[i][1]))
            if answer[i][1] != statment[i][1]: 
                result = False
        return result

    def give_score(self, answer):
        sequence = self.sentance_sequence(answer)

        solutions = [0,0,0,0,0,0,0,0,0,0,0]
        for seq in sequence:
            if seq[1]:
                for i in seq[1]:
                    if self.context(seq[0], i): 
                        solutions[i] = 1
                        break
        total = sum(solutions)

        return 3 if total > 3 else total
