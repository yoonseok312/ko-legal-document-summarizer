import pandas as pd
import os
import json

PROJECT_DIR = os.getcwd()
print(PROJECT_DIR)

DATA_DIR = f'{PROJECT_DIR}/data'
RAW_DATA_DIR = DATA_DIR + '/raw'

def load_data(mode: str):

    if mode == 'train':

        with open(f"./data/{mode}.json", "r", encoding='UTF-8-sig') as st_json:
            train = json.load(st_json)

            sent_list = []
            if_ext_list = []

            for article in train:
                ext_list = article['extractive']
                original = article['article_original']
                if_ext = [int(sent_num in ext_list) for sent_num in range(len(original))]

                sent_list += original
                if_ext_list += if_ext

            print(len(sent_list))

            sent_list = sent_list
            if_ext_list = if_ext_list

            train_data = pd.DataFrame(list(zip(sent_list, if_ext_list)),
                                      columns=['sentence', 'if_ext'])

            for c in ",.:;":
                train_data["sentence"] = train_data["sentence"].str.replace(c, "")

            for c in "()[]":
                train_data["sentence"] = train_data["sentence"].str.replace(c, " ")

            train_data.to_pickle(f"./data/{mode}_df.pickle")

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