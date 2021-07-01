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
from rnn_tokenize import tokenize
from model import RNNModel
from LSTM import LSTMModel

def make_submission():
    # Create RNN
    # input_dim = 512  # input dimension
    # hidden_dim = 1024  # hidden layer dimension
    # layer_dim = 5  # number of hidden layers
    # output_dim = 2  # output dimension
    # visible_gpus = 0
    # seq_len = 20

    # LSTM configs
    batch_size = 32
    n_iters = 50000
    visible_gpus = 0
    seed = 7777

    # Create RNN
    input_dim = 128  # input dimension
    hidden_dim = 256  # hidden layer dimension
    layer_dim = 4  # number of hidden layers
    output_dim = 2  # output dimension
    seq_len = 50



    device = "cpu" if visible_gpus == '-1' else f"cuda:{visible_gpus}"
    device_id = 0 if device == f"cuda" else -1

    # model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim, device)
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, device).to(device=device)
    model.load_state_dict((torch.load('./model/seq_len_40_2/model_26000.pth')))

    train_data = pd.read_pickle(f"./data/train_df.pickle")
    tokenized_data, tokenized_valid_data, embedding_model, target_train, target_test, l_tokenizer = tokenize(input_dim)

    word_extractor = WordExtractor()
    word_extractor.train(train_data['sentence'])
    # word_score_table = word_extractor.extract()

    with open("./data/test.json", "r", encoding='UTF-8-sig') as st_json:
        test = json.load(st_json)

    test_sent_list = []

    for article in test:
        original = article['article_original']
        test_sent_list += original

    test_data = pd.DataFrame(test_sent_list, columns=['sentence'])

    for c in ",.:;":
        test_data["sentence"] = test_data["sentence"].str.replace(c, "")

    for c in "()[]":
        test_data["sentence"] = test_data["sentence"].str.replace(c, " ")

    tokenized_test_data = [l_tokenizer.tokenize(sentence, flatten=True) for sentence in test_data['sentence']]

    test_list = []
    zero_list = [0] * input_dim
    for sent in tokenized_test_data:
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
        test_list += [temp_list]

    test_tensor = torch.from_numpy(np.array(test_list, dtype=np.float64)).float()
    test_placehorder = torch.from_numpy(np.array([0] * len(test_list), dtype=np.float64)).float()
    test_dataset = TensorDataset(test_tensor, test_placehorder)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("Loaded test dataset")

    output_list = []
    count = 0
    for i, (images, labels) in enumerate(test_loader):
        for_test = Variable(images.view(-1, seq_len, input_dim))

        if torch.cuda.is_available():
            for_test.to(device=f"cuda:{visible_gpus}")

        # Forward propagation
        output = model(for_test)
        output = torch.nn.functional.sigmoid(output)
        output_list.append(output[0][1].item())

    test_data["if_ext"] = output_list

    sub = []
    sent_count = 0

    for n, data in enumerate(test):
        article_original = data['article_original']
        id = data['id']
        result = []
        for sent_num, sent in enumerate(article_original):
            ext_pos = test_data.iloc[sent_count, 1]
            if len(result) >= 3:
                result = sorted(result, key=(lambda x: x[1]), reverse=True)
                if ext_pos > result[-1][1]:
                    result = result[:-1]
                    result.append((sent_num, ext_pos))
            else:
                result.append((sent_num, ext_pos))
            sent_count += 1
        if len(result) != 3:
            print(result)
        sorted(result, key=(lambda x: x[1]), reverse=True)
        sub.append(result)

    with open("./output/sample_submission.json", "r") as st_json:
        submission_template = json.load(st_json)

    for n in range(len(sub)):
        for i in range(3):
            submission_template[n]['summary_index' + str(i + 1)] = sub[n][i][0]

    with open("./output/lstm_seqlen40_2_sigmo.json", "w") as json_file:
        json.dump(submission_template, json_file)

if __name__ == '__main__':
    make_submission()