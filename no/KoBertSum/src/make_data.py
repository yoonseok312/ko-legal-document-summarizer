import os
import sys
import re
#import MeCab
from bs4 import BeautifulSoup
import kss
import json
import numpy as np 
import pandas as pd
from tqdm import tqdm
import argparse
import codecs
import pickle
import collections

PROBLEM = 'ext'

PROJECT_DIR = '..'

DATA_DIR = f'{PROJECT_DIR}/{PROBLEM}/data'
RAW_DATA_DIR = DATA_DIR + '/raw'
JSON_DATA_DIR = DATA_DIR + '/json_data'
BERT_DATA_DIR = DATA_DIR + '/bert_data' 
LOG_DIR = f'{PROJECT_DIR}/{PROBLEM}/logs'
LOG_PREPO_FILE = LOG_DIR + '/preprocessing.log' 

def number_split(sentence):
    num_str_pattern = re.compile(r'(\s\d+)([^\d\s])')
    sentence = re.sub(num_str_pattern, r'\1 \2', sentence)

    sentence_fixed = ''
    for token in sentence.split():
        if token.isnumeric():
            token = ' '.join(token)
        sentence_fixed+=' '+token
    return sentence_fixed

def noise_remove(text):
    text = text.lower()

    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(' +', ' ', text)
    text = re.sub('¶', ' ', text)
    text = re.sub('----------------', ' ', text)
    text = re.sub(';', '.', text)

    return text

def preprocessing(text, tokenizer=None):
    text = noise_remove(text)
    if tokenizer is not None:
        text = tokenizer(text)
        text = ' '.join(text)

    return text

def korean_sent_spliter(doc):
    sents_splited = kss.split_sentences(doc)
    if len(sents_splited) == 1:
        return sents_splited
    else:  
        for i in range(len(sents_splited) - 1):
            idx = 0
            if sents_splited[idx][-1] not in ['.','?' ] and idx < len(sents_splited) - 1:
                sents_splited[idx] = sents_splited[idx] + ' ' + sents_splited[idx + 1] if doc[len(sents_splited[0])] == ' ' \
                                        else sents_splited[idx] + sents_splited[idx + 1] 
                del sents_splited[idx + 1]
                idx -= 1
        return sents_splited


def create_json_files(df, data_type='train', target_summary_sent=None, path=''):
    if data_type == 'valid':
        NUM_DOCS_IN_ONE_FILE = 2500
    else:
        NUM_DOCS_IN_ONE_FILE = 1000
    start_idx_list = list(range(0, len(df), NUM_DOCS_IN_ONE_FILE))

    for start_idx in tqdm(start_idx_list):
        end_idx = start_idx + NUM_DOCS_IN_ONE_FILE
        if end_idx > len(df):
            end_idx = len(df)  

        length = len(str(len(df)))
        start_idx_str = (length - len(str(start_idx)))*'0' + str(start_idx)
        end_idx_str = (length - len(str(end_idx-1)))*'0' + str(end_idx-1)

        file_name = os.path.join(f'{path}/{data_type}_{target_summary_sent}' \
                                + f'/{data_type}.{start_idx_str}_{end_idx_str}.json') if target_summary_sent is not None \
                    else os.path.join(f'{path}/{data_type}' \
                                + f'/{data_type}.{start_idx_str}_{end_idx_str}.json')
        
        json_list = []
        for i, row in df.iloc[start_idx:end_idx].iterrows():
            original_sents_list = [preprocessing(original_sent).split()  
                                    for original_sent in row['article_original']]

            summary_sents_list = []
            if target_summary_sent is not None:
                if target_summary_sent == 'ext':
                    summary_sents = row['extractive_sents']
                elif target_summary_sent == 'abs':
                    summary_sents = korean_sent_spliter(row['abstractive'])   
                summary_sents_list = [preprocessing(original_sent).split() 
                                        for original_sent in summary_sents]

            json_list.append({'src': original_sents_list,
                              'tgt': summary_sents_list
            })
        json_string = json.dumps(json_list, indent=4, ensure_ascii=False)
        with open(file_name, 'w') as json_file:
            json_file.write(json_string)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", default=None, type=str, choices=['df', 'train_bert', 'test_bert'])
    parser.add_argument("-target_summary_sent", default='abs', type=str)
    parser.add_argument("-n_cpus", default='2', type=str)

    args = parser.parse_args()

    # python make_data.py -make df
    # Convert raw data to df
    if args.task == 'df': # and valid_df
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(RAW_DATA_DIR, exist_ok=True)

        # import data
        # with open(f'{RAW_DATA_DIR}/train.json', 'r') as json_file:
        train_json_list = json.load(codecs.open(f'{RAW_DATA_DIR}/train.json', 'r', 'utf-8-sig'))
        # with open(f'{RAW_DATA_DIR}/test.json', 'r') as json_file:
        #     test_json_list = list(json_file)
        test_json_list = json.load(codecs.open(f'{RAW_DATA_DIR}/test.json', 'r', 'utf-8-sig'))

        # Convert raw data to df
        df = pd.DataFrame(train_json_list)
        df['extractive_sents'] = df.apply(lambda row: list(np.array(row['article_original'])[row['extractive']]) , axis=1)
        
        
        #split test, validation set based on distributions of article length with ratio given (default = 0.9)
        ratio = 0.9
        id_by_length = collections.defaultdict(list)
        total_dataset_size = len(df)
        train_ids = []
        valid_ids = []


        for i in range(total_dataset_size):
            id_by_length[len(df.iloc[i]['article_original'])].append(df.iloc[i]['id'])

        for length, idxs in id_by_length.items():
            split_point = int(len(idxs) * ratio)
            train_ids += idxs[0:split_point]
            valid_ids += idxs[split_point:]

            
        train_df = df.loc[df['id'].isin(train_ids)]
        valid_df = df.loc[df['id'].isin(valid_ids)]

        train_df.reset_index(inplace=True, drop=True)
        valid_df.reset_index(inplace=True, drop=True)

        test_df = pd.DataFrame(test_json_list)

        # save df
        train_df.to_pickle(f"{RAW_DATA_DIR}/train_df.pickle")
        valid_df.to_pickle(f"{RAW_DATA_DIR}/valid_df.pickle")
        test_df.to_pickle(f"{RAW_DATA_DIR}/test_df.pickle")
        print(f'train_df({len(train_df)}) is exported')
        print(f'valid_df({len(valid_df)}) is exported')
        print(f'test_df({len(test_df)}) is exported')
        
    # python make_data.py -make bert -by abs
    # Make bert input file for train and valid from df file
    elif args.task  == 'train_bert':
        os.makedirs(JSON_DATA_DIR, exist_ok=True)
        os.makedirs(BERT_DATA_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

        for data_type in ['train', 'valid']:
            df = pd.read_pickle(f"{RAW_DATA_DIR}/{data_type}_df.pickle")

            ## make json file
            # 동일한 파일명 존재하면 덮어쓰는게 아니라 ignore됨에 따라 폴더 내 삭제 후 만들어주기
            json_data_dir = f"{JSON_DATA_DIR}/{data_type}_{args.target_summary_sent}"
            if os.path.exists(json_data_dir):
                os.system(f"rm {json_data_dir}/*")
            else:
                os.mkdir(json_data_dir)

            create_json_files(df, data_type=data_type, target_summary_sent=args.target_summary_sent, path=JSON_DATA_DIR)
           
            ## Convert json to bert.pt files
            bert_data_dir = f"{BERT_DATA_DIR}/{data_type}_{args.target_summary_sent}"
            if os.path.exists(bert_data_dir):
                os.system(f"rm {bert_data_dir}/*")
            else:
                os.mkdir(bert_data_dir)
            
            os.system(f"python preprocess.py"
                + f" -mode format_to_bert -dataset {data_type}"
                + f" -raw_path {json_data_dir}"
                + f" -save_path {bert_data_dir}"
                + f" -log_file {LOG_PREPO_FILE}"
                + f" -lower -n_cpus {args.n_cpus}")


    # python make_data.py -task test_bert
    # Make bert input file for test from df file
    elif args.task  == 'test_bert':
        os.makedirs(JSON_DATA_DIR, exist_ok=True)
        os.makedirs(BERT_DATA_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

        test_df = pd.read_pickle(f"{RAW_DATA_DIR}/test_df.pickle")

        ## make json file
        # 동일한 파일명 존재하면 덮어쓰는게 아니라 ignore됨에 따라 폴더 내 삭제 후 만들어주기
        json_data_dir = f"{JSON_DATA_DIR}/test"
        if os.path.exists(json_data_dir):
            os.system(f"rm {json_data_dir}/*")
        else:
            os.mkdir(json_data_dir)

        create_json_files(test_df, data_type='test', path=JSON_DATA_DIR)
        
        ## Convert json to bert.pt files
        bert_data_dir = f"{BERT_DATA_DIR}/test"
        if os.path.exists(bert_data_dir):
            os.system(f"rm {bert_data_dir}/*")
        else:
            os.mkdir(bert_data_dir)
        
        os.system(f"python preprocess.py"
            + f" -mode format_to_bert -dataset test"
            + f" -raw_path {json_data_dir}"
            + f" -save_path {bert_data_dir}"
            + f" -log_file {LOG_PREPO_FILE}"
            + f" -lower -n_cpus {args.n_cpus}")
