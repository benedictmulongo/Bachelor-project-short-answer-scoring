# https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html
import nltk
import gensim
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models

facit_file = 'doc1_light.txt'

class lda:

    def __init__(self):
        self.stopWords = set(stopwords.words('english'))
        self.punctuation = ['.',')','(','/','=','*','¨','%','€','|','&',',']
        #self.p_stemmer = PorterStemmer()

        # Import file and put it in a list
        with open(facit_file, 'r') as file:
            raw = file.read().splitlines()

        doc_clean = [self._clean(doc) for doc in raw]
        #print(doc_clean)

        self.dictionary = corpora.Dictionary(doc_clean)
        self.corpus = [self.dictionary.doc2bow(text) for text in doc_clean]
        self.create_model()

    def _clean(self, data):
        tokens = nltk.word_tokenize(data)
        stopped = [i for i in tokens if not i in self.stopWords]
        stopped = [w for w in stopped if w not in self.punctuation]
        #stopped = [self.p_stemmer.stem(i) for i in stopped]
        return stopped

    def create_model(self, num_topics=7, passes=100):
        self.model = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=num_topics, id2word = self.dictionary, passes=passes)

    def give_topic(self, question):
        topics = self._use_model(self.model, question)
        topic, prob = max(topics, key=lambda x:x[1])
        return topic #if prob>0.25 else 0

    def give_score(self, answer):
        return self.give_topic(answer)

    def _use_model(self, model, question, mini=None):
        question = nltk.word_tokenize(question)
        question = [i for i in question if not i in self.stopWords]
        question = [w for w in question if w not in self.punctuation]
        #question = [self.p_stemmer.stem(i) for i in question]
        question = self.dictionary.doc2bow(question)
        return model.get_document_topics(question, minimum_probability=mini)

    def diff(self, answer):
        clean = self._clean(answer)
        dictionary = corpora.Dictionary([clean])
        corpus = list(dictionary.doc2bow(clean))
        ma = gensim.models.ldamodel.LdaModel(corpus, num_topics=7, id2word = dictionary, passes=100)
        return self.model.diff(ma)


