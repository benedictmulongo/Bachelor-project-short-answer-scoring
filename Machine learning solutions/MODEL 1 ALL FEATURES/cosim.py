#https://stevenloria.com/tf-idf/
#http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

#X = 'In order to perform the experiment the way the students did one thing you would need to know is how much vinegar to put in the container. Also what size container should be used. Another thing that should be known is what temperature should the samples kept at. Lastly what temperature should the water use to rinse off the samples be and how long should they be rinsed.'

# Y = 'I would frustred to know where is being tested another word the specif materials which we are taking samples of in addition i would need to know the type of vinegar being since different vinegar have different ph along with the amount of vinegar added to each container these three pieces of information are all reason to replicate the experiment.'

# class cosine:

#     def __init__(self):
#         self.tfidf_vectorizer = TfidfVectorizer()
#         self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(X)
#         #print(self.tfidf_matrix[0:1]

reference_file = 'refanswer.txt'
with open(reference_file, 'r') as file:
    references = file.read().splitlines()


def cos_give_score(answer):
    references.insert(0, answer)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(tuple(references))
    co_sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)[0]
    return max(co_sims[1:])
