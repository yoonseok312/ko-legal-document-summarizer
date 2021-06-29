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
    test_data = pd.read_pickle(f"./data/test_df.pickle")

    train_sent = train_data['sentence']
    test_sent = test_data['sentence']

    all_sent = pd.concat([train_sent, test_sent])

    word_extractor = WordExtractor()
    word_extractor.train(all_sent)
    word_score_table = word_extractor.extract()

    scores = {word: score.cohesion_forward for word, score in word_score_table.items()}
    l_tokenizer = LTokenizer(scores=scores)
    tokenized_all_data = [l_tokenizer.tokenize(sentence, flatten=True) for sentence in all_sent]
    tokenized_train_data = [l_tokenizer.tokenize(sentence, flatten=True) for sentence in train_sent]
    # tokenized_test_data = [l_tokenizer.tokenize(sentence, flatten=True) for sentence in test_data['sentence']]
    # tokenized_all_data = tokenized_train_data + tokenized_test_data

    embedding_model = Word2Vec(sentences=tokenized_all_data, vector_size=input_dim, window=8, min_count=1, workers=16, sg=0)

    return tokenized_train_data, embedding_model, l_tokenizer

def create_dataset(tokenized_data: List, embedding_model: Word2Vec, input_dim: int, seq_len: int):

    train_data = pd.read_pickle(f"./data/train_df.pickle")
    input_list = []
    for sent in tqdm(tokenized_data):
        temp_list = []
        sent = sent[-seq_len:]
        for word_count, word in enumerate(sent):
            if word_count == seq_len:
                break
            if word in embedding_model.wv:
                temp_list += list(rating for rating in embedding_model.wv[word])
            else:
                temp_list += [0] * input_dim
        temp_list = [0] * input_dim * (seq_len - len(sent)) + temp_list
        if len(temp_list) != input_dim * seq_len:
            print('hell')
        input_list.append(temp_list)

    target_list = list(train_data['if_ext'])

    input_train, input_test, target_train, target_test = train_test_split(np.array(input_list),
                                                                              np.array(target_list),
                                                                              test_size=0.2,
                                                                              random_state=42)
    return input_list, input_train, input_test, target_train, target_test