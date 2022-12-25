import pandas as pd
import re
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.isri import ISRIStemmer

# from emoji import demojize
# from translate import Translator
# translator= Translator(to_lang="Arabic")

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

emojisTranslation = {
    '♀': 'انثى',
    '💉': 'حقنه',
    '🌹': 'ورد',
    '🌷': 'ورد',
    '✅': 'صحيح',
    '✔':'صحيح',
    '😀': 'يضحك',
    '😁': 'يضحك',
    '😂': 'يضحك',
    '😅': 'محرج',
    '💔': 'قلب مكسور',
    '🔴': 'تحذير',
    '⭕': 'تحذير',
    '📌': 'تحذير',
    '📍': 'تحذير',
    '❗':'تحذير',
    '⛔':'تحذير',
     '⚠':'تحذير',
    '😍': 'اعجاب',
    '😎': 'فخر',
    '💪': 'قوه',
    '👍': 'حسنا',
    '🙄': 'استغراب',
    '👌': 'ممتاز',
    '✌': 'نصر',
    '✋': 'يد',
    '💚': 'قلب',
     '❤': 'قلب',
     '💙': 'قلب',
     '💛':'قلب',
     '📸':'كاميرا',
     '🎥': 'كاميرا',
    '👇': 'اسفل',
    '😉': 'غمزه',
    '😜': 'غمزه',
    '😔': 'محبط',
    '😭': 'يبكي',
    '♂':'ذكر',
     '✍':'يكتب',
     '😷': 'كمامه',
}

handmadePatterns = [
r'وفاه' ,
r'وفاه.*لقاح' ,
r'عاجل' ,      
r'وزاره الصحه' ,
r'رفض' ,
r'كورونا' ,
r'يتلقى' ,
r'لقاح' ,
r'(استرازينيكا)|(موديرنا)|(سبوتنيك)|(فايزر)|(بيونتيك)|(سينوفارم)' ,
r'لقاح امن',
r'تطعيم' ,
]

def preprocessDF(filepath:str, type:str, getNERandPOS=False, applyStemming=False):
    '''
    The function gets the file path of csv file, produces new file (type_processed.csv) after preprocessing for exploring

    Inputs
    filepath: path of the input csv file
    type: train/test/etc.. for output name type_processed.csv
    getNERandPOS: OPTIONAL (default False) boolean value if we want to return NER and POS and add it to output file
    applyStemming: OPTIONAL (default False) to apply nltk arabic stemming

    Return
    data_x: list of tokenized sentences
    handmadeFeature: list of vectors of handamade features
    data_y: list of labels [category, stance]
    ner_data: (ONLY IF getNERandPOS is true)
    pos_data: (ONLY IF getNERandPOS is true)
    '''
    data_x = []
    handmadeFeatures = []
    pos_data = []
    ner_data = []

    data_y = []

    df = pd.read_csv(filepath)

    ### SOME TWEET CLEANING
    df['text']=df['text'].str.lower() #English words to lower case

    for emojiKey in emojisTranslation.keys(): #REPLACE EMOJIS WITH MEANINGS
        df['text'].replace(to_replace = emojiKey, value =' '+emojisTranslation[emojiKey]+' ', regex = True, inplace=True) 


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
    st = None if not applyStemming else ISRIStemmer()


    for index, item in df.iterrows():
        tweet = item['text']
        tokenizedTweet = []
        tweet_tokens = tokenizer.tokenize(tweet)
        for word in tweet_tokens:
            if (word not in arabicStopwords):  # remove stopwords
                tokenizedTweet.append(word)

        if getNERandPOS:
            ner_data.append(ner.predict_sentence(tokenizedTweet))
            disambig = mle.disambiguate(tokenizedTweet)
            tweet_pos = [d.analyses[0].analysis['pos'] for d in disambig]
            pos_data.append(tweet_pos)

        featureVector = [1 if re.search(handmadePatterns[i],item['text'])!=None else 0 for i in range(len(handmadePatterns))] #Handmade feature vector
        handmadeFeatures.append(featureVector)

        if applyStemming: #Applying stemming if needed
            tokenizedTweet = [st.stem(word) for word in tokenizedTweet]

        df.at[index,'text'] = ' '.join(tokenizedTweet)
        df.at[index,'category'] = categoriesMap[item['category']]

        data_x.append(tokenizedTweet)
        data_y.append([categoriesMap[item['category']], item['stance']]) #Change categories to numbers

    if getNERandPOS:
        df.insert(1,"NER", ner_data)
        df.insert(1,"POS", pos_data)
    
    for i in range(len(handmadePatterns)):
        df.insert(1+i,"feature"+str(i), [item[i] for item in handmadeFeatures])

    df.to_csv(type+'_processed.csv', index=False)

    if getNERandPOS:
        return data_x, handmadeFeatures, data_y, ner_data, pos_data
    else:
        return data_x, handmadeFeatures, data_y
    

def demojizeArabic(emo):
  res=demojize(emo, delimiters=("", ""))
  english = res.replace("_"," ")
  if len(english) < 3:
    return ''
  translation = translator.translate(english)
  return translation

def getMostFrequentEmojisTranslation(data, countThreshold:int):
    emojiPattern = r'[\u263a-\U0001f645]'
    usedEmojis = set()
    emojisCount = {}
    emojisMap = {}
    for index, item in data.iterrows():
        if re.search(emojiPattern,item['text'])!=None:
            regex = re.compile(emojiPattern)
            #print(regex.findall(item['text']))
            for emoji in set(regex.findall(item['text'])):
                usedEmojis.add(emoji)
                if emoji not in emojisCount.keys():
                    emojisCount[emoji] = 1
                else:
                    emojisCount[emoji] += 1
    keysToBeRemoved = []
    for key in emojisCount.keys():
        if emojisCount[key]< countThreshold:
            keysToBeRemoved.append(key)

    for key in keysToBeRemoved:
        usedEmojis.remove(key)

    for emoji in usedEmojis:
        emojisMap[emoji] = demojizeArabic(emoji)
    
    return emojisMap

