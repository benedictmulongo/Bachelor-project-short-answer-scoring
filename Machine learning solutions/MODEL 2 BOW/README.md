
# Model 2
# Python implementation and how to use

Use the function in the file ml_bow_model_features_computation.py to compute all features implemented. The function compute_features() computes all features and return a matrix of all feature vector. The function evaluate model compute X and y from the data returne by compute_features and use that to evaluate the model using logistic regression implemented in SKLEARN. 

# Features implementation 

dictionary = 
[
  "mater",
  "how much",
  "distilled",
  "water",
  "sample",
  "distilled",
  "peice",
  "vineger",
  "size",
  "put",
  ...
  ].
  
Two very simple features are implemented. The first feature computes a vector representation of each answer using a predifined dictionary. The feature vector represents the occurence of each word in the dictionary. The classfier used is logistic regression. 
