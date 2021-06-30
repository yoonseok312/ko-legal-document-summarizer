import pandas as pd
import os
import json
import random
import pickle
import collections

PROJECT_DIR = os.getcwd()
print(PROJECT_DIR)

DATA_DIR = f'{PROJECT_DIR}/data'
RAW_DATA_DIR = DATA_DIR + '/raw'

def split_data():
    with open(f"./data/train.json", "r", encoding='UTF-8-sig') as st_json:
        train = json.load(st_json)

        df = pd.DataFrame(train)

        ratio = 0.9
        id_by_length = collections.defaultdict(list)
        total_dataset_size = len(df)
        train_ids = []
        valid_ids = []

        for i in range(total_dataset_size):
            id_by_length[len(df.iloc[i]['article_original'])].append(df.iloc[i]['id'])

        for length, idxs in id_by_length.items():
            split_point = int(len(idxs) * (1 - ratio))
            valid_ids += idxs[0:split_point]
            train_ids += idxs[split_point:]

        train_df = df.loc[df['id'].isin(train_ids)]
        valid_df = df.loc[df['id'].isin(valid_ids)]

        train_df.reset_index(inplace=True, drop=True)
        valid_df.reset_index(inplace=True, drop=True)

        # save df
        train_df.to_pickle("./data/train_unprocessed_df.pickle")
        valid_df.to_pickle("./data/valid_unprocessed_df.pickle")

def load_data(mode: str):

    if mode == 'train':

        train = pd.read_pickle("./data/train_unprocessed_df.pickle")

        train_sent_list = []
        train_ext_list = []

        for index, article in train.iterrows():
            ext_list = article['extractive']
            original = article['article_original']
            if_ext = [int(sent_num in ext_list) for sent_num in range(len(original))]

            train_sent_list += original
            train_ext_list += if_ext

        train_data = pd.DataFrame(list(zip(train_sent_list, train_ext_list)),
                                  columns=['sentence', 'if_ext'])

        for c in ",.:;":
            train_data["sentence"] = train_data["sentence"].str.replace(c, "")

        for c in "()[]":
            train_data["sentence"] = train_data["sentence"].str.replace(c, " ")

        train_data.to_pickle(f"./data/{mode}_df.pickle")

    elif mode == 'valid':
        valid = pd.read_pickle("./data/valid_unprocessed_df.pickle")

        valid_sent_list = []
        valid_ext_list = []
        valid_ext_list_hit = []

        for index, article in valid.iterrows():
            ext_list = article['extractive']
            original = article['article_original']
            if_ext = [int(sent_num in ext_list) for sent_num in range(len(original))]

            valid_sent_list += original
            valid_ext_list += if_ext
            valid_ext_list_hit.append((ext_list, len(original)))

        valid_data = pd.DataFrame(list(zip(valid_sent_list, valid_ext_list)),
                                  columns=['sentence', 'if_ext'])

        for c in ",.:;":
            valid_data["sentence"] = valid_data["sentence"].str.replace(c, "")

        for c in "()[]":
            valid_data["sentence"] = valid_data["sentence"].str.replace(c, " ")

        valid_data.to_pickle(f"./data/valid_df.pickle")
        with open("./data/valid_ext_list_hit", "wb") as f:
            pickle.dump(valid_ext_list_hit, f)

    else:
        with open(f"./data/{mode}.json", "r", encoding='UTF-8-sig') as st_json:
            train = json.load(st_json)

            sent_list = []

            for article in train:
                original = article['article_original']

                sent_list += original

            print(len(sent_list))

            sent_list = sent_list

            test_data = pd.DataFrame(list(sent_list),
                                      columns=['sentence'])

            for c in ",.:;":
                test_data["sentence"] = test_data["sentence"].str.replace(c, "")

            for c in "()[]":
                test_data["sentence"] = test_data["sentence"].str.replace(c, " ")

            test_data.to_pickle(f"./data/{mode}_df.pickle")
if __name__ == '__main__':
    split_data()
    load_data('train')
    load_data('valid')
    load_data('test')