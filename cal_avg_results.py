import numpy as np
from pprint import pprint


FILE_TYPE = 'test'
MODEL_TYPE = 'xlmr'
READ_NAME = f'outputs/panx/xlm-roberta-base_{MODEL_TYPE}_LR2e-5-epoch10-MaxLen128/{FILE_TYPE}_results.txt'

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
        for i in range(2, 6):
            item = lang[i]
            item_val = item.split('=')[1].strip()
            lang_ret.append(item_val)
        results.append(lang_ret)
    # pprint(results)

    # calculate the average over all languages
    array_ret = np.array(results, dtype=float)
    print('f1', 'loss', 'precision', 'recall')
    print(np.average(array_ret, 0))

