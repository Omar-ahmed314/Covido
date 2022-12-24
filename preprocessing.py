import pandas as pd
import re
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

# from camel_tools.ner import NERecognizer
# from camel_tools.disambig.mle import MLEDisambiguator

categoriesMap = {
    'info_news':0,
    'celebrity':1,
    'plan':2,
    'requests':3,
    'rumors':4,
    'advice':5,
    'restrictions':6,
    'personal':7,
    'unrelated': 8,
    'others':9,
}

def preprocessDF(filepath:str, type:str, getNERandPOS=False):
    '''
    The function gets the file path of csv file, produces new file (type_processed.csv) after preprocessing for exploring

    Inputs
    filepath: path of the input csv file
    type: train/test/etc.. for output name type_processed.csv
    getNERandPOS: OPTIONAL (default False) boolean value if we want to return NER and POS and add it to output file

    Return
    data_x: list of tokenized sentences
    data_y: list of labels [category, stance]
    ner_data: (ONLY IF getNERandPOS is true)
    pos_data: (ONLY IF getNERandPOS is true)
    '''
    data_x = []
    pos_data = []
    ner_data = []

    data_y = []

    df = pd.read_csv(filepath)

    ### SOME TWEET CLEANING
    df['text']=df['text'].str.lower() #English words to lower case
    df['text'].replace(to_replace =r'http[\da-zA-Z:/.-]*\b', value = '', regex = True, inplace=True) #Removing URLs
    df['text'].replace(to_replace =r'<lf>', value = ' ', regex = True, inplace=True) #Remove tags
    df['text'].replace(to_replace =r'#', value = '', regex = True, inplace=True) #Remove hashtags
    df['text'].replace(to_replace =r'_', value = ' ', regex = True, inplace=True) #Replace underscores
    df['text'].replace(to_replace =r'ـ', value = '', regex = True, inplace=True) #Remove reduntant word elongation عــــاجل => عاجل
    df['text'].replace(to_replace =r'@\w\b', value = '', regex = True, inplace=True) #Remove mentions
    df['text'].replace(to_replace =r'@', value = '', regex = True, inplace=True) #Remove @
    df['text'].replace(to_replace =r'user', value = '', regex = True, inplace=True) #Remove all occurences of USER
    df['text'].replace(to_replace =r'\b[أإآ]', value = 'ا', regex = True, inplace=True) #Change all forms of Alif at the beginning of word to one form (أسد) -> (اسد) #Not Sure if important
    df['text'].replace(to_replace =r'ة\b', value = 'ه', regex = True, inplace=True) #Assume people mix between ة and ه at end of words, we make it all ه #Not Sure if important
    df['text'].replace(to_replace =r'[ڤڨ]', value = 'ف', regex = True, inplace=True) #Replace ڤ with ف


    df['text'].replace(to_replace =r'covid(19)?(-19)?( 19)?( - 19)?', value = 'كورونا', regex = True, inplace=True) #Change all forms of covid words to one form (covid19, coronavirus, فيروس كورونا,...) to كورونا
    df['text'].replace(to_replace =r'corona(\s?virus)?', value = 'كورونا', regex = True, inplace=True) 
    df['text'].replace(to_replace =r'(فيرو?س )?(ال)?كورونا', value = 'كورونا', regex = True, inplace=True) 
    df['text'].replace(to_replace =r'كوفيد(19)?(-19)?( 19)?( - 19)?(١٩)?(-١٩)?( ١٩)?( - ١٩)?', value = 'كورونا', regex = True, inplace=True)

    df['text'].replace(to_replace =r'astrazeneca', value = 'استرازينيكا', regex = True, inplace=True) #Translate vaccine related words
    df['text'].replace(to_replace =r'moderna', value = 'موديرنا', regex = True, inplace=True)
    df['text'].replace(to_replace =r'sputnik( v)?', value = 'سبوتنيك', regex = True, inplace=True)
    df['text'].replace(to_replace =r'pfizer', value = 'فايزر', regex = True, inplace=True)
    df['text'].replace(to_replace =r'biontech', value = 'بيونتيك', regex = True, inplace=True)
    df['text'].replace(to_replace =r'sinopharm', value = 'سينوفارم', regex = True, inplace=True)


    df['text'].replace(to_replace =r'[^\w\s]', value = ' ', regex = True, inplace=True) #remove punctuation
    df['text'].replace(to_replace =r'[0-9٠١٢٣٤٥٦٧٨٩]', value = '', regex = True, inplace=True) #remove numbers
    df['text'].replace(to_replace =r'[a-z]', value = '', regex = True, inplace=True) #remove english letters
    df['text'].replace(to_replace =r'( )+', value = ' ', regex = True, inplace=True) #remove long spaces
    

    ### TOKENIZATION
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)

    arabicStopwords = stopwords.words('arabic')


    ner = None if not getNERandPOS else getNERecognizer.pretrained()
    mle = None if not getNERandPOS else MLEDisambiguator.pretrained()


    for index, item in df.iterrows():
        tweet = item['text']
        tokenizedTweet = []
        tweet_tokens = tokenizer.tokenize(tweet)
        for word in tweet_tokens:
            if (word not in arabicStopwords):  # remove stopwords
                tokenizedTweet.append(word)

        df.at[index,'text'] = ' '.join(tokenizedTweet)
        df.at[index,'category'] = categoriesMap[item['category']]

        if getNERandPOS:
            ner_data.append(ner.predict_sentence(tokenizedTweet))
            disambig = mle.disambiguate(tokenizedTweet)
            tweet_pos = [d.analyses[0].analysis['pos'] for d in disambig]
            pos_data.append(tweet_pos)

        data_x.append(tokenizedTweet)
        data_y.append([categoriesMap[item['category']], item['stance']]) #Change categories to numbers

    if getNERandPOS:
        df.insert(1,"NER", ner_data)
        df.insert(1,"POS", pos_data)

    df.to_csv(type+'_processed.csv', index=False)

    if getNERandPOS:
        return data_x, data_y, ner_data, pos_data
    else:
        return data_x,data_y
    
