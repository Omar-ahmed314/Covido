import pandas as pd
import re
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

def preprocessDF(filepath:str):
    '''
    The function gets the file path of csv file, produces new file (processed.csv) after preprocessing for exploring (includes stopwords)
    Return
    data_x: list of tokenized sentences
    data_y: list of labels
    '''
    data_x = []
    data_y = []
    #REMAINING TODO: Replace emojis with sentiment
    #REMAINING TODO: Stemming and lemmatization if needed

    df = pd.read_csv(filepath)

    #changing category labels to numbers
    df['category'] = pd.Categorical(df['category'], categories=df['category'].unique()).codes

    ### SOME TWEET CLEANING
    df['text'].replace(to_replace =r'http[\da-zA-Z:/.-]*\b', value = '', regex = True, inplace=True) #Removing URLs
    df['text'].replace(to_replace =r'<LF>', value = ' ', regex = True, inplace=True) #Remove tags
    df['text'].replace(to_replace =r'#', value = '', regex = True, inplace=True) #Remove hashtags
    df['text'].replace(to_replace =r'_', value = ' ', regex = True, inplace=True) #Replace underscores
    df['text'].replace(to_replace =r'ـ', value = '', regex = True, inplace=True) #Remove reduntant word elongation عــــاجل => عاجل
    df['text'].replace(to_replace =r'@\w\b', value = '', regex = True, inplace=True) #Remove mentions
    df['text'].replace(to_replace =r'@', value = '', regex = True, inplace=True) #Remove @
    df['text'].replace(to_replace =r'USER', value = '', regex = True, inplace=True) #Remove all occurences of USER
    df['text'].replace(to_replace =r'\b[أإآ]', value = 'ا', regex = True, inplace=True) #Change all forms of Alif at the beginning of word to one form (أسد) -> (اسد) #Not Sure if important
    df['text'].replace(to_replace =r'ة\b', value = 'ه', regex = True, inplace=True) #Assume people mix between ة and ه at end of words, we make it all ه #Not Sure if important
    df['text'].replace(to_replace =r'[ڤڨ]', value = 'ف', regex = True, inplace=True) #Replace ڤ with ف

    df['text'].replace(to_replace =r'[Cc][Oo][Vv][Ii][Dd](19)?(-19)?( 19)?( - 19)?', value = 'كورونا', regex = True, inplace=True) #Change all forms of covid words to one form (covid19, coronavirus, فيروس كورونا,...) to كورونا
    df['text'].replace(to_replace =r'[Cc][Oo][Rr][Oo][Nn][Aa](\s?[Vv][Ii][Rr][Uu][Ss])?', value = 'كورونا', regex = True, inplace=True) 
    df['text'].replace(to_replace =r'(فيرو?س )?(ال)?كورونا', value = 'كورونا', regex = True, inplace=True) 
    df['text'].replace(to_replace =r'كوفيد(19)?(-19)?( 19)?( - 19)?(١٩)?(-١٩)?( ١٩)?( - ١٩)?', value = 'كورونا', regex = True, inplace=True)

    df['text'].replace(to_replace =r'[^\w\s]', value = ' ', regex = True, inplace=True) #remove punctuation
    df['text'].replace(to_replace =r'( )+', value = ' ', regex = True, inplace=True) #remove long spaces
    df['text'].replace(to_replace =r'[0-9٠١٢٣٤٥٦٧٨٩]', value = '', regex = True, inplace=True) #remove numbers

    df['text'].to_csv('processed.csv')

    ### TOKENIZATION
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)

    arabicStopwords = stopwords.words('arabic')
    

    for index, item in df.iterrows():
        tweet = item['text']
        tokenizedTweet = []
        tweet_tokens = tokenizer.tokenize(tweet)
        for word in tweet_tokens:
            if (word not in arabicStopwords):  # remove stopwords
                tokenizedTweet.append(word)

        data_x.append(tokenizedTweet)
        data_y.append([item['category'], item['stance']])


    return data_x, data_y
    