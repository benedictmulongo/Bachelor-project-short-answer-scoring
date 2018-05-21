import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

facit = [   'How much vinegar', 'Size of container', 'Type of container',
            'Type of material','Type of samples','Size of material',
            'Size of samples','Type of vinegar','Location of container',
            'Surface area of sample','Room temperature','How to rinse',
            'container covered','samples completely submerged','Type of climate']

class Holistic:

    def __init__(self):
        self.stopWords = set(stopwords.words('english'))
        self.punctuation = ['.',',',')','(']
        self.stemmer = SnowballStemmer("english")
        self.pointers = [self.clean(i) for i in facit]

    def clean(self, text):
        text = text.lower()
        tokens = list(set(nltk.word_tokenize(text)))
        tokens = [i for i in tokens if i not in self.stopWords]
        tokens = [i for i in tokens if i not in self.punctuation]
        tokens = [self.stemmer.stem(i) for i in tokens]
        return tokens

    def give_score(self, answer):
        sentances = nltk.sent_tokenize(answer)
        clean_answer = [self.clean(i) for i in sentances]
        points = self.compare_points(clean_answer)
        return 3 if points > 3 else points

    def compare_points(self, student):
        count = 0
        for statment in self.pointers:
            for sentance in student:
                filled = 0
                for word in sentance:
                    if word in statment:
                        filled += 1
                if filled >= len(statment):
                    count += 1
        return count
