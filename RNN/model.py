import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, seq_len, device):
        super(RNNModel, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.seq_len = seq_len

        self.device = device

        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu').to(device=device)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.to(device)


    def forward(self, x):
        # Initialize hidden state with zeros
        # print("input x", x.shape)
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device))
        # print("input h", x.shape)

        x = x.to(device=self.device)

        # One time step
        out, hn = self.rnn(x, h0)

        # print("out", out.shape)
        # print("rnn output", out.shape)
        out = self.fc(out[:, -1, :])
        # print("linear out", out.shape)
        # out = self.fc(out)
        return out