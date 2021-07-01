import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from soynlp.tokenizer import LTokenizer
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from rnn_tokenize import tokenize, create_dataset
from model import RNNModel
from LSTM import LSTMModel

def hit_rate_stat():

    visible_gpus = 0
    seed = 777
    # Create RNN
    input_dim = 128  # input dimension
    hidden_dim = 256  # hidden layer dimension
    layer_dim = 1  # number of hidden layers
    output_dim = 2  # output dimension
    seq_len = 20

    device = "cpu" 
    device_id = -1

    # model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim, device)
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, device).to(device=device)
    model.load_state_dict(torch.load('./model/seq_len/model_26000.pth', map_location=torch.device('cpu')))

    tokenized_data, tokenized_valid_data, embedding_model,  train_data, valid_data, l_tokenizer = tokenize(input_dim)

    _, input_valid, _, target_valid = create_dataset(
        tokenized_data,
        tokenized_valid_data,
        embedding_model,
        input_dim,
        seq_len,
        train_data,
        valid_data
    )

    valid_list = []
    zero_list = [0] * input_dim
    for sent in tokenized_valid_data:
        temp_list = []
        sent = sent[-seq_len:]
        for word_count, word in enumerate(sent):
            if word_count == seq_len:
                break
            if word in embedding_model.wv:
                temp_list += [rating for rating in embedding_model.wv[word]]
            else:
                temp_list += zero_list
        temp_list = zero_list * (seq_len - len(sent)) + temp_list
        valid_list += [temp_list]

    valid_tensor = torch.from_numpy(np.array(valid_list, dtype=np.float64)).float()
    valid_placehorder = torch.from_numpy(np.array([0] * len(valid_list), dtype=np.float64)).float()
    valid_dataset = TensorDataset(valid_tensor, valid_placehorder)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    print("Loaded valid dataset")

    seq_dim = 20
    output_list = []
    count = 0
    for i, (images, labels) in enumerate(valid_loader):
        for_test = Variable(images.view(-1, seq_dim, input_dim))

        if torch.cuda.is_available():
            for_test.to(device=f"cuda:{visible_gpus}")

        # Forward propagation
        output = model(for_test)
        output = torch.nn.functional.softmax(output, dim=1)
        output_list.append(output.data[0][1])

    valid_data["if_ext"] = output_list

    sub = []
    sent_count = 0

    with open(f"./data/valid.json", "r", encoding='UTF-8-sig') as st_json:
        valid = json.load(st_json)

    hit_rate_stat_dict = {}

    def dict_update(key, sent_len, t_dict, value):
        if sent_len in t_dict.keys(): 
            if key in t_dict[sent_len].keys():
                t_dict[sent_len][key] += value
            else:
                t_dict[sent_len][key] = value
        else:
            t_dict[sent_len] = {key: value}

    for n, data in enumerate(valid):
        article_original = data['article_original']
        ext = data['extractive']
        
        result = []
        for sent_num, sent in enumerate(article_original):
            ext_pos = valid_data.iloc[sent_count, 1].item()
            if len(result) >= 3:
                result = sorted(result, key=(lambda x: x[1]), reverse=True)
                if ext_pos > result[-1][1]:
                    result = result[:-1]
                    result.append((sent_num, ext_pos, len(sent)))
            else:
                result.append((sent_num, ext_pos, len(sent)))
            sent_count += 1
        if len(result) != 3:
            print(result)
        


        for sent_num, _, sent_len in result:
            if sent_num in ext:
                dict_update('correct', sent_len, hit_rate_stat_dict, 1)
            dict_update('correct', sent_len, hit_rate_stat_dict, 0)
            dict_update('total', sent_len, hit_rate_stat_dict, 1)

    for k, v in hit_rate_stat_dict.items():
        print('sent_len: {}   acc: {}%   count: {}'.format(k, v['correct']/v['total']*100, v['total']))

if __name__ == '__main__':
    hit_rate_stat()