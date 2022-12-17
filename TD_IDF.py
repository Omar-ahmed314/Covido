import pandas as pd
import re
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from preprocssing import preprocessDF , get_vocab
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

def tfidf_transformation( tweets):
    #instantiate CountVectorizer() 
    cv=CountVectorizer() 
    # this steps generates word counts for the words in your docs 
    #create tf matrix of size (N*V) 
    word_count_vector=cv.fit_transform(tweets)
    print(word_count_vector.shape)
    #idf(t) = log [ n / df(t) ] + 1  
    # smoothed ->, terms that occur in all documents in a training set, will not be entirely ignored.
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
    tfidf_transformer.fit(word_count_vector) # TODO Fit wla fit transform
    # print idf values 
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"]) 
    # sort ascending 
    df_idf.sort_values(by=['idf_weights'])
    # compute TF for each tweet that generate Matrix of size (N*V) 
    # each row represent a tweet, each col a word the cell has count of this word in that tweet
    count_vector=cv.transform(tweets) 
    #print(count_vector.shape)
    # tf-idf scores using TF matrix and IDFs valued that we fit the tfidf_transformer object using it
    tf_idf=tfidf_transformer.transform(count_vector)
    return tf_idf
