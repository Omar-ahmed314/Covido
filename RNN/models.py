import torch as t

class Module(t.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
        super(Module, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_dim
        self.embedding = t.nn.Embedding(vocab_size, embedding_dim=embedding_dim)
        self.fc = t.nn.Linear(hidden_size, num_classes)
    
    def forward(self, input):
        pass

class RNN(Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
        super().__init__(vocab_size, embedding_dim, hidden_size, num_layers, num_classes)
        self.rnn = t.nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        
    def forward(self, input):
        input = self.embedding(t.tensor(input))
        out, _ = self.rnn(input)
        out = self.fc(out)
        return out
        
class LSTM(Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
        super().__init__(vocab_size, embedding_dim, hidden_size, num_layers, num_classes)
        self.lstm = t.nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        
    def forward(self, input):
        input = self.embedding(t.tensor(input))
        out, _ = self.lstm(input)
        out = self.fc(out)
        return out
