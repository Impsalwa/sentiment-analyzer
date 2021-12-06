# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 16:53:25 2021

@author: Salwa
"""

from future.utils import iteritems
import nltk 
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup #for XML parser (the data is XML files)

#initilize the lemmatizer 
wordnet_lemmatizer = WordNetLemmatizer() #turns words into their base form 
#remove the stopwords
stopwords = set(w.rstrip() for w in open("stopwords.txt"))
#open and read the data 
positive_reviews = BeautifulSoup(open(r'positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')
#print(positive_reviews)

#same with neagtive reveiws 
negative_reviews = BeautifulSoup(open(r'negative.review').read())
negative_reviews = negative_reviews.findAll('review_text')
#print(negative_reviews)

np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]

#create an index for each word in the final data vector 
#we need to know the size of the data vector by knowing how big is the vocabulary
#create a dictionary for the vocabulary 
def my_tokenizer(s):
    s = s.lower() # lowercase 
    tokens = nltk.tokenize.word_tokenize(s) #toknize the data 
    tokens = [t for t in tokens if len(t) > 2] #keep only the words more then 2 charecters 
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]#lemmatize the words 
    tokens  = [t for t in tokens if t not in stopwords] #removing stopwords
    return tokens
word_index_map ={}
current_index = 0 #counter increase when a new word shows 
#save the tokenized data to use later 
positive_tokenized = []
negative_tokenized = []

for review in positive_reviews:
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1
            
#same thing for the negative reviews
for reviw in negative_reviews:
    tokens = my_tokenizer(reviw.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1
            
#take each token and create a data array (a vector of a token)
def token_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1) # last element is for the label
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x/ x.sum()
    x[-1] = label
    return x 

N = len(positive_tokenized) + len(negative_tokenized)

data = np.zeros((N, len(word_index_map) +1))
i = 0
for tokens in positive_tokenized:
    xy = token_to_vector(tokens, 1)
    data [i:] =xy
    i +=1
    
for tokens in negative_tokenized:
    xy = token_to_vector(tokens, 0)
    data[i:] = xy
    i += 1
    
 
np.random.shuffle(data)
#prepare train and test data
X = data[:,:-1] #all rows except last column
Y = data[:,-1] 
#print (X)
#print(Y)
#split the train and test data
xtrain = X[:-100,]
ytrain = Y[:-100,]
# last 100 rows will be test
xtest = X[-100:,]
ytest = Y[-100:,]
#start classification 
model= LogisticRegression()
model.fit(xtrain,ytrain)
score_train = model.score(xtrain, ytrain)
score = model.score(xtest, ytest)

print("Train score ", score_train)
print("Test score : ", score)

#to increase the rate score 
# let's look at the weights for each word
#for those far from 0 
#try threshold with deffrent values 
threshold = 0.5
for word, index in iteritems(word_index_map):
    weight = model.coef_[0][index]
    if weight > threshold or weight < -threshold:
        print(word, weight)#some of the results are logic but some are not 
        


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



















