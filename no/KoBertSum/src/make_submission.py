import json
import numpy as np
import pandas as pd
import time
import codecs
import re
import sys
import os

PROBLEM = 'ext'

## 사용할 path 정의
# PROJECT_DIR = '/home/uoneway/Project/PreSumm_ko'
PROJECT_DIR = '..'
print(PROJECT_DIR)

DATA_DIR = f'{PROJECT_DIR}/{PROBLEM}/data'
RAW_DATA_DIR = DATA_DIR + '/raw'
JSON_DATA_DIR = DATA_DIR + '/json_data'
BERT_DATA_DIR = DATA_DIR + '/bert_data' 
LOG_DIR = f'{PROJECT_DIR}/{PROBLEM}/logs'
LOG_PREPO_FILE = LOG_DIR + '/preprocessing.log' 

MODEL_DIR = f'{PROJECT_DIR}/{PROBLEM}/models' 
RESULT_DIR = f'{PROJECT_DIR}/{PROBLEM}/results' 

# python make_submission.py result_1209_1236_step_7000.candidate
if __name__ == '__main__':
    # test set
    json_list = json.load(codecs.open(f'{RAW_DATA_DIR}/test.json', 'r', 'utf-8-sig'))

    test_df = pd.DataFrame(json_list)
    test_df = test_df['id']

    # 추론결과
    with open(RESULT_DIR + '/' + f'result_1209_1236_step_8000_num.csv', 'r') as file:
        print("arg:", sys.argv[1])
        lines = file.readlines()
    # print(lines)
    test_pred_list = []
    cnt = 0
    for line in lines:
        top_3 = list(map(int, line.split(',')))
        print(top_3)
        while len(top_3) < 3:
            for i in range(0, 10):
                if i not in top_3:
                    cnt += 1
                    top_3.append(i)
                    break

        test_pred_list.append({
            'summary_index1': top_3[0],
            'summary_index2': top_3[1],
            'summary_index3': top_3[2],
        })
    print(cnt)

    # pred_list_df = pd.DataFrame(test_pred_list)
    # pred_list_df['id'] = test_df

    result_df = pd.merge(test_df, pd.DataFrame(test_pred_list), how="left", left_index=True, right_index=True)
    # result_df['summary'] = result_df.apply(lambda row: '\n'.join(list(np.array(row['article_original'])[row['sum_sents_idxes']])) , axis=1)

    result_df['id'] = result_df['id'].astype(int)
    result_df = result_df.rename(columns={"id":"ID"})
    result_json = result_df.to_json(orient='records')

    print(result_df['ID'].dtypes)



    ## 결과 통계치 보기
    # word
    # abstractive_word_counts = submit_df['summary'].apply(lambda x:len(re.split('\s', x)))
    # print(abstractive_word_counts.describe())

    # export
    now = time.strftime('%y%m%d_%H%M')
    result_df.to_csv(f'{RESULT_DIR}/submission_{now}.csv', index=False, encoding="utf-8-sig")

    with open(f'{RESULT_DIR}/submission_{now}.json', 'w') as json_file:
        json_file.write(str(result_json))
