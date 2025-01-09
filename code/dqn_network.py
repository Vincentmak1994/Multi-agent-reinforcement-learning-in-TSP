import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_observation, n_actions, hidden_size_1=128, hidden_size_2=256, dropout_rate=0.2):
        super().__init__()
        self.location_size = 2 #x and y coordinates 
        self.layer1 = nn.Linear(n_observation+self.location_size, hidden_size_1)
        self.layer2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.out = nn.Linear(hidden_size_2, n_actions)

    def forward(self, t):
        t = F.relu(self.layer1(t))
        t = F.relu(self.layer2(t))
        return self.out(t)
    
class DQN_RNN(nn.Module):
    def __init__(self, num_cities, hidden_size_1=64, hidden_size_2=128):
        super().__init__()
        embedding_size = 32
        self.location_size = 2 #x and y coordinates 
        self.input_layer = nn.Linear(self.location_size, embedding_size)
        self.hidden_layer1 = nn.Linear(embedding_size, hidden_size_1)
        self.hidden_layer2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.rnn = nn.LSTM(hidden_size_2, hidden_size_2, batch_first=True)
        self.fc = nn.Linear(hidden_size_2, num_cities)
    
    def forward(self, x):
        output = F.relu(self.input_layer(x))
        output = F.relu(self.hidden_layer1(output))
        output = F.relu(self.hidden_layer2(output))
        output, _ = self.rnn(output)
        output = self.fc(output)
        return output 