import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def BOW(df):
    '''
    The function gets the dataframe and returns the BOW vector of the text column
    '''
    bow= CountVectorizer()
    bow_vector= bow.fit_transform(df['text'])
    return bow_vector 

def TFIDF(df):
    '''
    The function gets the dataframe and returns the TFIDF vector of the text column
    '''
    tfidf= TfidfVectorizer()
    tfidf_vector= tfidf.fit_transform(df['text'])
    return tfidf_vector 


def word2vec(df):
    '''
    The function gets the dataframe and returns the word2vec vector of the text column
    '''
    tknzr = TweetTokenizer()
    df = df['text'].apply(lambda x: tknzr.tokenize(x))
    model_train = Word2Vec(df, window=5, min_count=1, workers=4)
    model_train.train(df, total_examples= len(df), epochs=20)

    #create a vector for each tweet
    def word_vector(tokens, size):
        vector = np.zeros(size).reshape((1,size))
        count = 0
        for word in tokens:
            try:
                vector += model_train.wv[word].reshape(1,size)
                count += 1
            except KeyError:
                pass
        if count != 0:
            vector /= count
        return vector

    #apply vectorize_tweet to df
    wordvec_arrays = np.zeros((len(df), 100)) 
    for i in range(len(df)):
        wordvec_arrays[i,:] = word_vector(df[i], 100)

    wordvec_df = pd.DataFrame(wordvec_arrays)
    
    return wordvec_df
