import numpy as np
from pprint import pprint
import pandas as pd


FILE_TYPE = 'test'
MODEL_TYPE = 'xlmr-mh'
TRAIN_LANG = 'en,de,fr'
PREDICT_HEAD = 'fr'
UNLABEL_DATA = 'ru,es,ja'
# READ_NAME = f'outputs/panx/xlm-roberta-base_{MODEL_TYPE}_TL{TRAIN_LANG}_PH{PREDICT_HEAD}_LR2e-5-epoch10-MaxLen128-uniform/{FILE_TYPE}_results.txt'
# READ_NAME = f'outputs/panx/xlm-roberta-base_xlmr-p_TLen,de,fr_PHmean_LR2e-5-epoch10-MaxLen128-uniform/{FILE_TYPE}_results.txt'
# READ_NAME = f'outputs/panx/xlm-roberta-base_xlmr-p_TLen,de,fr_PL8/{FILE_TYPE}_results.txt'
# READ_NAME = f'outputs/panx/xlm-roberta-base_xlmr-p_TLen,de,fr_PHmean_LR2e-5-epoch10-MaxLen128-uniform/{FILE_TYPE}_results.txt'
# READ_NAME = f'outputs/panx/xlm-roberta-base_xlmr-p_TLen,de,fr_ULru,es,ja,it,zh,ko,arLR2e-5-epoch10-MaxLen128/{FILE_TYPE}_results.txt'
READ_NAME = f'outputs/panx/xlm-roberta-base_xlmr-p_TLen,de,fr_ULru,es,jaLR2e-5-epoch10-MaxLen128-soft/{FILE_TYPE}_results.txt'
print(READ_NAME)

# read files
with open(READ_NAME, 'r') as f:
    langs = []
    cur_lang = []

    # collect each language strings into a python list
    for line in f.readlines():
        line = line.strip('\n')

        if line == '=====================' and cur_lang:
            langs.append(cur_lang)
            cur_lang = []

        cur_lang.append(line)
    langs.append(cur_lang)
    # pprint(langs)

    # clean the strings
    results = []
    for lang in langs:
        lang_ret = []
        for i in range(1, 6):
            item = lang[i]
            item_val = item.split('=')[1].strip()
            lang_ret.append(item_val)
        results.append(lang_ret)
    # pprint(results)

    # save to csv file.
    df = pd.DataFrame(results)
    df.columns = ['Lang', 'F1', 'Loss', 'Precision', 'Recall']
    df['F1'] = df['F1'].astype(float)
    df['Loss'] = df['Loss'].astype(float)
    df['Precision'] = df['Precision'].astype(float)
    df['Recall'] = df['Recall'].astype(float)
    df.loc[len(df)] = ['avg', df.loc[:, 'F1'].mean(), df.loc[:, 'Loss'].mean(), df.loc[:, 'Precision'].mean(), df.loc[:, 'Recall'].mean()]
    # df.to_csv(f'{MODEL_TYPE}_TL{TRAIN_LANG}_PH{PREDICT_HEAD}_UL{UNLABEL_DATA}.csv', float_format='%.4f', index=False)
    df.to_csv(f'mlf-p-3-soft.csv', float_format='%.4f', index=False)




