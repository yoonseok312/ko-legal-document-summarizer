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

def make_submission():
    # Create RNN
    input_dim = 100  # input dimension
    hidden_dim = 256  # hidden layer dimension
    layer_dim = 5  # number of hidden layers
    output_dim = 2  # output dimension
    visible_gpus = 0

    device = "cpu" if visible_gpus == '-1' else f"cuda:{visible_gpus}"
    device_id = 0 if device == f"cuda" else -1

    model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim, device)
    model.load_state_dict((torch.load('./model/model_5200.pth')))

    train_data = pd.read_pickle(f"./data/train_df.pickle")
    tokenized_data, embedding_model, l_tokenizer = tokenize()

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
    for sent in tokenized_test_data:
        temp_list = []
        sent = sent[-20:]
        for word_count, word in enumerate(sent):
            if word_count == 20:
                break
            if word in embedding_model.wv:
                temp_list += list(rating for rating in embedding_model.wv[word])
            else:
                temp_list += [0] * 100
        temp_list = [0] * 100 * (20 - len(sent)) + temp_list
        if len(temp_list) != 2000:
            print('hell')
        test_list.append(temp_list)

    test_tensor = torch.from_numpy(np.array(test_list, dtype=np.float64)).float()
    test_placehorder = torch.from_numpy(np.array([0] * len(test_list), dtype=np.float64)).float()
    test_dataset = TensorDataset(test_tensor, test_placehorder)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("Loaded test dataset")

    seq_dim = 20
    output_list = []
    count = 0
    for i, (images, labels) in enumerate(test_loader):
        for_test = Variable(images.view(-1, seq_dim, input_dim))

        # Forward propagation
        output = model(for_test)
        output = torch.nn.functional.softmax(output, dim=1)
        output_list.append(output.data[0][1])

    test_data["if_ext"] = output_list

    sub = []
    sent_count = 0

    for n, data in enumerate(test):
        article_original = data['article_original']
        id = data['id']
        result = []
        for sent_num, sent in enumerate(article_original):
            ext_pos = test_data.iloc[sent_count, 1].item()
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

    with open("./output/submission_06290222.json", "w") as json_file:
        json.dump(submission_template, json_file)

if __name__ == '__main__':
    make_submission()