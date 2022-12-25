import torch as t
import preprocessing
from torch.utils.data import Dataset

class TweetsDataset(Dataset):
    '''Generic dataset'''
    def __init__(self, csvPath, type):
        '''initialize the data and labels'''
        super().__init__()
        self.data, _, self.labels = preprocessing.preprocessDF(csvPath, type, getNERandPOS=False, applyStemming=True)
        self.vocabs = self.calculate_vocabs()
        self.vocab_size = len(self.vocabs)
        self.data_padding()
        self.replace_with_vocab_id()
        self.data = t.tensor(self.data)

    def __getitem__(self, index):
        ''' get the row item'''
        return self.data[index], self.labels[index]

    def data_padding(self):
        max_length = self.get_max_data_length()
        for data_row in self.data:
            if len(data_row) < max_length:
                padding = max_length - len(data_row)
                padding_list = [' '] * padding
                data_row += padding_list

    def get_max_data_length(self):
        max_length = -1e9
        for data_row in self.data:
            if len(data_row) > max_length:
                max_length = len(data_row)
        return max_length

    def calculate_vocabs(self):
        '''Calculate the vocabs book from the data'''
        vocabs_set = set()
        vocab_book = {}
        for tweet in self.data:
            vocabs_set.update(tweet)
        # here I started with token 1 as to leave zero token to the unknown word
        for i, word in enumerate(vocabs_set):
            vocab_book[word] = i + 1
        vocab_book['UNK'] = 0
        return vocab_book

    def replace_with_vocab_id(self):
        '''Replace the string word with its id in vocabs book'''
        for i, sentence in enumerate(self.data):
            self.data[i] = [self.vocabs[word] if word in self.vocabs else self.vocabs['UNK'] for word in sentence]


    def __len__(self):
        '''get the dataset size'''
        return len(self.data)
        

class CategoriesDataset(TweetsDataset):
    '''Tweets dataset with their categories'''
    def __init__(self, csvPath, type):
        '''initialize the data and labels'''
        super().__init__(csvPath, type)
        self.labels = t.tensor(self.labels)[:, 0]
        

class StancesDataset(TweetsDataset):
    '''Tweets dataset with their stances'''
    def __init__(self, csvPath, type):
        '''initialize the data and labels'''
        super().__init__(csvPath, type)
        self.labels = t.tensor(list(map(lambda x: 2 if x[1] == -1 else x[1], self.labels)))