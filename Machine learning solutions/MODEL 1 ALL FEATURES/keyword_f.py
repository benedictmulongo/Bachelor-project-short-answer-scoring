import nltk
from nltk.corpus import stopwords
import sys
sys.path.append('..')
import data_handler as dh

class Keyword:

    def __init__(self):
        self.stopWords = set(stopwords.words('english'))
        self.punctuation = ['.',')','(','/','=','*','¨','%','€','|','&',',']
        self.keywords = self._generate_keywords()

    def _generate_keywords(self):
        lines = dh.get_facit(1)
        return [self._give_synonyms(line) for line in lines]

    def _give_synonyms(self, reference):
        tokens = list(set(nltk.word_tokenize(reference)))
        tokens = [i for i in tokens if i not in self.stopWords]
        tokens = [i for i in tokens if i not in self.punctuation]

        synonyms = list()
        from nltk.corpus import wordnet as wn
        from itertools import chain
        for token in tokens:
            sets = wn.synsets(token)
            #lemmas = list(set(chain.from_iterable([syn.lemma_names() for syn in sets])))
            lemmas = sets[0].lemma_names()
            synonyms.append(lemmas)    

        return synonyms

    def count_keywords(self, answer, multiplier=1):
        words = nltk.word_tokenize(answer)
        uniq_words = list(set(words))
        uniq_words = [i for i in uniq_words if i not in self.stopWords]
        uniq_words = [i for i in uniq_words if i not in self.punctuation]

        result = [self._count_key(uniq_words, ref) for ref in self.keywords]
        result = sorted(result, reverse=True)[:3]
        count = sum(result)
        return count#/(len(words)*multiplier)

    def give_score(self, answer):
        return self.count_keywords(answer)

    def _count_key(self, answer, keywords):
        count = 0
        for token in answer:
            for lemmas in keywords:
                if token in lemmas:
                    count += 1
                    break
        return count

# if __name__ == "__main__":
#     ke = Keyword()
#     print(*ke.keywords, sep='\n')
#     print( ke.count_keywords("time is not on our end for this pee") )