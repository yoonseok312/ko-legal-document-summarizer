import torch
from torch.utils.data import Dataset

class SentenceDataset(Dataset):
    def __init__(self, input_data, target_data, embedding_model, seq_len, input_dim):
        self.input_data = input_data
        self.target_data = target_data
        self.embedding_model = embedding_model
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.length = len(input_data)

    def __getitem__(self, index):
        #create sentence vector using embedding_model
        sent_input_vec = []
        zero_list = [0] * self.input_dim
        sent = (self.input_data[index])[-self.seq_len:]
        for word in sent:
            if word in self.embedding_model.wv:
                sent_input_vec += [rating for rating in self.embedding_model.wv[word]]
            else:
                sent_input_vec += zero_list
        sent_input_vec = zero_list * (self.seq_len - len(sent)) + sent_input_vec
        return torch.tensor(sent_input_vec), torch.tensor(self.target_data[index])

    def __len__(self):
        return self.length

