import pandas as pd
import os
import json
import random
import pickle

PROJECT_DIR = os.getcwd()
print(PROJECT_DIR)

DATA_DIR = f'{PROJECT_DIR}/data'
RAW_DATA_DIR = DATA_DIR + '/raw'

def load_data(mode: str):

    if mode == 'train':

        with open(f"./data/{mode}.json", "r", encoding='UTF-8-sig') as st_json:
            train = json.load(st_json)

            train_sent_list = []
            valid_sent_list = []
            train_ext_list = []
            valid_ext_list = []
            valid_ext_list_hit = []

            for article in train:
                ext_list = article['extractive']
                original = article['article_original']
                if_ext = [int(sent_num in ext_list) for sent_num in range(len(original))]

                if random.random() > 0.8:
                    valid_sent_list += original
                    valid_ext_list += if_ext
                    valid_ext_list_hit.append((ext_list, len(original)))
                else:
                    train_sent_list += original
                    train_ext_list += if_ext

            train_data = pd.DataFrame(list(zip(train_sent_list, train_ext_list)),
                                      columns=['sentence', 'if_ext'])
            valid_data = pd.DataFrame(list(zip(valid_sent_list, valid_ext_list)),
                                      columns=['sentence', 'if_ext'])

            for c in ",.:;":
                train_data["sentence"] = train_data["sentence"].str.replace(c, "")
                valid_data["sentence"] = valid_data["sentence"].str.replace(c, "")

            for c in "()[]":
                train_data["sentence"] = train_data["sentence"].str.replace(c, " ")
                valid_data["sentence"] = valid_data["sentence"].str.replace(c, " ")

            train_data.to_pickle(f"./data/{mode}_df.pickle")
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
    load_data('train')
    load_data('test')