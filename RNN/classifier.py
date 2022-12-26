import torch as t
from torch.utils.data import DataLoader
from RNN.models import RNN, LSTM

class TweetsClassifier():
    def __init__(self, dataset, embedding_dim, hidden_size, num_layers, num_classes, epoch_size, learning_rate):
        super().__init__()
        self.dataset = dataset
        self.embedding_dim = embedding_dim
        self.epoch_size = epoch_size
        self.learning_rate = learning_rate
        self.data_loader = DataLoader(dataset=dataset, batch_size=256, shuffle=True)
        self.criterion = t.nn.CrossEntropyLoss()

    def train(self):
        for epoch in range(self.epoch_size):
            total_model_acc = 0
            total_num_samples = 0
            for data in self.data_loader:
                train_data = data[:][0]
                labels = data[:][1]
                output = self.model(train_data)
                total_model_acc += (t.argmax(output, dim=2) == labels.repeat(output.shape[1], 1).transpose(0, 1)).sum()
                total_num_samples += len(labels) * output.shape[1]
                loss = self.criterion(t.reshape(output, (output.shape[0] * output.shape[1], -1)), labels.repeat(output.shape[1], 1).transpose(0, 1).flatten())
                self.optimizer.zero_grad()
                loss.backward()
                t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()
            
            total_model_acc = total_model_acc / total_num_samples
            print(f"epoch num: {epoch} has accuracy: {100 * total_model_acc}")

    def predict(self, testLoader):
        with t.no_grad():
            predected = t.tensor([])
            real_labels = t.tensor([])
            for tweets in testLoader:
                test_data = tweets[:][0]
                labels = tweets[:][1]
                output = self.model(test_data)
                predected = t.cat((predected, t.argmax(output[:, -1, :], dim=1)))
                real_labels = t.cat((real_labels, labels))
            return predected, real_labels


class RnnTweetsClassifier(TweetsClassifier):
    def __init__(self, dataset, embedding_dim, hidden_size, num_layers, num_classes, epoch_size, learning_rate):
        super().__init__(dataset, embedding_dim, hidden_size, num_layers, num_classes, epoch_size, learning_rate)
        self.model = RNN(dataset.vocab_size, embedding_dim, hidden_size, num_layers, num_classes)
        self.optimizer = t.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
        

class LSTMTweetsClassifier(TweetsClassifier):
    def __init__(self, dataset, embedding_dim, hidden_size, num_layers, num_classes, epoch_size, learning_rate):
        super().__init__(dataset, embedding_dim, hidden_size, num_layers, num_classes, epoch_size, learning_rate)
        self.model = LSTM(dataset.vocab_size, embedding_dim, hidden_size, num_layers, num_classes)
        self.optimizer = t.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)