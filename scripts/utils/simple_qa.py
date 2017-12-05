import pandas as pd

# Simple QA Dataset statistics
# Num Test Rows: 21687
# Num Dev Rows: 10845
# Num Train Rows: 75910
# Total Rows: 108442

simple_qa = {
    'train': '../../data/SimpleQuestions_v2/annotated_fb_data_train.txt',
    'dev': '../../data/SimpleQuestions_v2/annotated_fb_data_valid.txt',
    'test': '../../data/SimpleQuestions_v2/annotated_fb_data_test.txt',
}


def preprocess(row):
    # Remove `www.freebase.com/` to match format of PostgreSQL
    row['subject'] = row['subject'].strip().replace('www.freebase.com/m/', '')
    row['relation'] = row['relation'].strip().replace('www.freebase.com/', '')
    row['object'] = row['object'].strip().replace('www.freebase.com/m/', '')
    return row


def load_simple_qa(dev=False, train=False, test=False):
    ret = []
    for is_load, filename in [(train, simple_qa['train']), (dev, simple_qa['dev']),
                              (test, simple_qa['test'])]:
        if is_load:
            df = pd.read_table(
                filename, header=None, names=['subject', 'relation', 'object', 'question'])
            df = df.apply(preprocess, axis=1)
            ret.append(df)
    return tuple(ret)