
# Model 1
# Python implementation and how to use

bla bla 

# Theory behind each features

# Approach

Using machine learning for short answer scoring, we want to find pattern in the dataset that can help us predict the score of other student answers not in the training dataset. The only thing available is the student answers text and the corresponding score. 
As stated before machine learning algorithms work with numerical values  , the main difficulty is to implement algorithms to transform the student answer to a numerical value that can be feed into a machine learning algorithm. To achieve this transformation from free text to numerical values, 19 features or characteristics of the text are implemented. Those features takes the text as input and return a numerical value representing a distance measure between the text and a reference answer, a similarity score, a number of keywords etc. 
# Feature Implementation
Features are characteristics of the text. Those characteristics can help us know if the text is worth 3 points or 0 points depending of the numerical value of the feature, also the attribute. A simple example of a feature is how many keyworks occur in the student text. A detailed explanation of these features can be found in [2], [7], [10], [36]–[42]. The following features were implemented and computed for each answer.
# Cosine Similarity
The simplest form of the cosine similarity uses the bag-of-words representation of two text and compute the distance between their respective vectors using the following :
simularity =( V_1 〖*V〗_2)/( | V_1 || V_2 |)
The similarity score between the vector representation of the reference texts and the student answers were used as a feature in the project. 
# Keywords
Keywords is the simplest feature, where we have a list of each keywords and a simply count of many times each keywords occur in the student answers. 
We have here used two versions of the keywords features, one that simples count the occurence of each keywords and return the result as a feature and another one that normalized the result. Those two features are then saved in the feature matrix as explained above.
# Latent Semantic Analysis
A complete explanation of the latent semantic analysis will be lengthy and out of the scope of this thesis. Nonetheless the latent semantic analysis can be defined as a method to grasp the latent (hidden) semantic space of two or more texts even if they do not necessarily share the same words. 
Given the dictionary [we, need, to, know, measure, the, quantity, amount, of, vinegar] and four vector representation of four text using this dictionary we can build the following matrix :
[ [1, 1, 1, 1],
  [1, 1, 0, 3],
  [1, 1, 1, 1],
  [1, 0, 1, 0],
  [0, 1, 0, 1],
  [1, 1, 1, 1],
  [3, 0, 1, 0],
  [0, 1, 0, 1],
  [1, 1, 1, 2],
  [1, 1, 1, 1] ]
Each column is a transposed vector representation of each text using the bag-of-words approach with a common dictionary. 
After the construction of the matrix we calculate the singular value decomposition  of the matrix X :
SDV(X)=UΣV^Tthe obtained matrices is reduced to a lower dimension to find a approximation and in the same time reducing the computation time needed as SDV(X_k)= U_k Σ_k V_k^Tis in a lower dimension. 
The terms are represented by U_k Σ_k and the text (documents) by Σ_k V_k^T, the similarity between two terms or document in the semantic space can be computed by using the cosine similarity between the two vectors. 
Two versions of the LSA model have been implemented as features, one that uses one reference answer where each sentence creditworthy is considered as document. The feature is calculated by computing averaging the similarity score between the student answer and each sentence in the reference answer. 
Another LSA model is build with a set of hundred reference answers as document and the highest similarity between the student answer and the reference answers set is return as a feature. 
# Partial Word Overlap
Partial word overlap is a method to compare the text by compute the word overlap between them by the following formula.
PCW(A,B) =(|A∩B|)/(|A+B|)   
We use the word partial here because it not a complete matching of words between the two texts in comparaison, but an approximate matching where we allow some difference between each word for example vinegar and vinegar will be matched with an approximate string matching. 
# Language Model
The language model is a method commonly used in automatic word suggestions or completion for example when typing in Google or other search engines. 
They calculate the probability of the next coming words given the preceding observed word. for example 
P(lose weight | how to ) or P(draw|how to)
When typing in google ‘how to’, the phrase ‘lose weight’ and ‘draw’ appear among the words showing up first. That means in Google there is a higher probability that peoples usually search ‘lose weight’ or ‘draw’ after typing ‘how to’. To estimate those probability, we need a very large corpus that is representative for the goal or application considered. To reduce null probability, the probability is approximate to unigram and the indepence of words is sometimes assumed. 
We have here make a large corpus of acceptable answers and calculate the perplexity of each student answer as a feature, that is the probability that the answer is from the corpus given the answer text. 
# Latent Dirichlet Algorithm
Latent dirichlet allocation is a very advanced algorithm using both higher probability model distribution and advanced methods as gibbs sampling because the correct estimate of the probabilities used in the model is NP-hard. 
We use the algorithm implemented in gensim  to implement the LDA features. Two versions are implemented.
The first version use the references answers and make a LDA model of it. Then the student answers is reduced to the same dimension as the references answers and the topic distribution probability of the student answers is used as a feature. 
# Word Alignment
Word alignment is a way to compare two text by calculating how many words between the two given text share the same semantic signification. It reminds in many ways to the partial word overlap, but here we are only interested to the word to word semantic analysis not necessary their syntactical structure. 
The formula to calculate the word alignment is :
sim(S^((1)),S^((2))) =  (n_c^a (S^((1))) + n_c^a (S^((2))) )/(n_c (S^((1))) + n_c (S^((2)))    )     
S^((1))  and S^((2)) are two input texts
n_c = number of content word in the input text without stop words (all word counts)
n_c^a = number of word in the input text aligned (without stopwords )
# Corpus Similarity
Corpus similarity use a set of keywords from the reference answer and look for each such word in the student answer, if not founded the algorithm looks up synonyms of the words and again match it against the student answer. The number of matched word is used as a feature. 
# Jaccard
Jaccard similarity is used to calculate the similarity between two vector representation of text using the following formula :
JACCARD(A,B) =(|A∩B|)/(|A∪B|)   
The similarity between the student answers and each reference answers is calculated and the average score is return as a feature. 
# Dice Similarity
Dice similarity calculate the similarity between two vector representation of text using the following formula :
DSC(A,B) =(2*|A∩B|)/(|A||B|)   
The similarity between the student answers and each reference answers is calculated and the highest score is return as a feature. 
Blue Score
Bleu score is a machine language translation method that can be used to estimate the quality of a machine translation. This quality estimate is calculated by comparing the machine translation against a set of reference human translations. The score is way to benchmark the performance of the machine translation and its quality. There have been recent research and attempts to modify the blue scoring algorithm in order to fit different needs for example in textual entailment or recently in essay scoring [36]. In order to use bleu as features , we have both used a unmodified blue algorithm from the natural language toolkit  and a modified blue algorithm based on [36] although some steps in the paper have been disregarded or modified. 
# Ngram
Solution set is divided into several statements witch will give points. Every statement is tested independently. Using python ngram search as a weight. The simultaneous weight of all the words in the answer compared to the given statement. After having done this on all the statements the three highest statements are combined and returned as a numerical value.
# Key
The feature that is extracted is the number of unique keywords found by comparing the answer to the selected keywords. The keywords are selected by modifying the solutions. Modifications is done by removing punctuations, stopwords and repeats, the words that are left are put into a list ki this is done for all the solutions. The solutions are then put into a list K.
	K = {k_1,k_2...k_i}
The extraction compares the student answer to each element in the list K. The comparison is done by counting how many of the keywords in ki was found. The results are stored in a list S.
	S = {s_1,s_2...s_i}		s_i=[0,|k_i |]
The resulting numerical return is the sum of the three highest valued si in S.
# Bingo
The solutions are divided up into smaller statements. This is done by hand by selecting the nouns and the corresponding help words to it. Each statement is put into a list. Stemming is used on all words in this method to get a higher comparison rate. Each statement is tested independently against the student answer. If the student answer have all the words of the statement in one sentence the student answer is given a point. After having gone through all the statements the return value is the sum of all points, if the sum exceeds 3 the return value is set to 3. Three statements are shown in Figure 7.


# Feature Extraction
Features extraction refers here to the computation of the features for each student answer in the training set. Each features mentioned above were computed for every student answers in order to build a matrix that a machine learning algorithm can used to make prediction. 
For each student answer we have computed a vector represented as follows :
x→feature(student)=[〖 f〗_1,f_2,f_3  ...f_16  ]
y→score(student) = range[0,3]
The list of each features vector computed and their corresponding scores is represented as matrix X and the labels as  y, forming together our training set.



