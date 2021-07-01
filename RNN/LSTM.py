import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(LSTMModel, self).__init__()

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim,
                            batch_first=True).to(device=device)
        # self.generator = Generator(hidden_dim, output_dim)
        # batch_first=True (batch_dim, seq_dim, feature_dim)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.device = device


    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        x = x.to(device=self.device)
        out, (hn, cn) = self.lstm(x, (h0.to(device=self.device).detach(), c0.to(device=self.device).detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out = self.generator(out[:, -1, :])
        # out.size() --> 100, 10
        return out



class Generator(nn.Module):
    def __init__(self, d_model, output_dim, hp=None):
        super().__init__()
        self.fc_1 = nn.Linear(d_model, int(d_model / 2))
        self.fc_2 = nn.Linear(int(d_model / 2), int(d_model / 4))
        self.fc_3 = nn.Linear(int(d_model / 4), output_dim)

        self.ln_1 = nn.LayerNorm(int(d_model / 2))
        self.ln_2 = nn.LayerNorm(int(d_model / 4))

    def forward(self, x):
        h = self.fc_1(x)
        h = self.ln_1(h)
        h = F.relu(h)

        h = self.fc_2(h)
        h = self.ln_2(h)
        h = F.relu(h)

        h = self.fc_3(h)
        h = torch.sigmoid(h)
        return h
