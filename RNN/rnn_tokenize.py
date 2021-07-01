from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm

import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List
import json


def tokenize(input_dim: int):

    train_data = pd.read_pickle(f"./data/train_df.pickle")
    valid_data = pd.read_pickle(f"./data/valid_df.pickle")
    test_data = pd.read_pickle(f"./data/test_df.pickle")

    train_sent = train_data['sentence']
    valid_sent = valid_data['sentence']
    test_sent = test_data['sentence']

    all_data = pd.concat([train_data, valid_data, test_data])

    all_sents = all_data['sentence']

    print(all_sents[0])
    train_sents = []
    valid_sents = []
    test_sents = []


    for i in range(len(train_data)):
        for j in  range(len(train_data['sentence'][i])):
            # print(train_data['sentence'][i][j])
            train_sents += [train_data['sentence'][i][j]]

    for i in range(len(valid_data)):
        for j in range(len(valid_data['sentence'][i])):
            valid_sents += [valid_data['sentence'][i][j]]

    for i in range(len(test_data)):
        for j in range(len(test_data['sentence'][i])):
            test_sents += [test_data['sentence'][i][j]]

    all_sent = train_sents + valid_sents + test_sents

    # print(all_sent)


    word_extractor = WordExtractor()
    word_extractor.train(all_sent)
    word_score_table = word_extractor.extract()

    scores = {word: score.cohesion_forward for word, score in word_score_table.items()}
    l_tokenizer = LTokenizer(scores=scores)
    tokenized_all_data = []
    tokenized_train_data = []
    tokenized_valid_data = []
    tokenized_for_vector = []
    # for index in range(len(all_sents)):
        # for sentence in all_sents[index]:
            # print(sentence)
            # break
    for article in all_data['sentence']:
        temp = []
        for sentence in article:
            tokenized_for_vector = [l_tokenizer.tokenize(sentence, flatten=True) for sentence in article]
            # temp += [l_tokenizer.tokenize(sentence, flatten=True)]
        # tokenized_all_data += [temp]
    print("all data done")
    for article in train_data['sentence']:
        temp = []
        for sentence in article:
            temp += [l_tokenizer.tokenize(sentence, flatten=True)]
        tokenized_train_data += [temp]
    for article in valid_data['sentence']:
        temp = []
        for sentence in article:
            temp += [l_tokenizer.tokenize(sentence, flatten=True)]
        tokenized_valid_data += [temp]
    # for index in range(len(train_sent)):
    #     article = train_data['sentence'][index].values[0]
    #     tokenized_train_data += [l_tokenizer.tokenize(sentence, flatten=True) for sentence in article]
    # for index in range(len(valid_sent)):
    #     article = valid_data['sentence'][index].values[0]
    #     tokenized_valid_data += [l_tokenizer.tokenize(sentence, flatten=True) for sentence in article]
    # tokenized_train_data = [l_tokenizer.tokenize(sentence, flatten=True) for sentence in train_sent]
    # tokenized_valid_data = [l_tokenizer.tokenize(sentence, flatten=True) for sentence in valid_sents]
    # tokenized_test_data = [l_tokenizer.tokenize(sentence, flatten=True) for sentence in test_data['sentence']]
    # tokenized_all_data = tokenized_train_data + tokenized_test_data

    embedding_model = Word2Vec(sentences=tokenized_for_vector, vector_size=input_dim, window=8, min_count=1, workers=16, sg=0)

    return tokenized_train_data, tokenized_valid_data, embedding_model, train_data, valid_data, l_tokenizer

def create_dataset(
        tokenized_train_data: List,
        tokenized_valid_data: List,
        embedding_model: Word2Vec,
        input_dim: int,
        seq_len: int,
        train_data,
        valid_data
):

    print("Start")
    input_train = []
    zero_list = [0] * input_dim
    max_sent_in_article = 50
    for article in tqdm(tokenized_train_data):
        article_list = []
        for sent in article:
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
            article_list += [temp_list]
        if len(article_list) > max_sent_in_article:
            print("error")
        while len(article_list) < max_sent_in_article:
            article_list += [[0] * input_dim * seq_len]
        input_train += [article_list]


    input_valid = []
    for article in tqdm(tokenized_valid_data):
        article_list = []
        for sent in article:
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
            article_list += [temp_list]
        if len(article_list) > max_sent_in_article:
            print("error")
        while len(article_list) < max_sent_in_article:
            article_list += [[0] * input_dim * seq_len]
        input_valid += [article_list]

    print("loop end")

    target_train = []
    pad_mask_train = []
    for item in list(train_data['if_ext']):
        temp = []
        # while len(item) < 50:
        target_train += [item + [-1] * (50 - len(item))]
        pad_mask_train += [[1] * len(item) + [0] * (50 - len(item))]
        # print("target", len(target_train))
        # print("pad", len(pad_mask_train))

    target_valid = []
    pad_mask_valid = []
    for item in list(valid_data['if_ext']):
        # while len(item) < 50:
        target_valid += [item + [-1] * (50 - len(item))]
        pad_mask_valid += [[1] * len(item) + [0] * (50 - len(item))]

    # target_train = list(train_data['if_ext'])
    # target_valid = list(valid_data['if_ext'])

    # input_train, input_test, target_train, target_test = train_test_split(np.array(input_list),
    #                                                                           np.array(target_list),
    #                                                                           test_size=0.2,
    #                                                                           random_state=42)

    print("dataset end")
    # print("tgt", target_train.shape)
    # print("pad", pad_mask_train.shape)
    return input_train, input_valid, target_train, target_valid, pad_mask_train, pad_mask_valid