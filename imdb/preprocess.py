import os
import glob
import pandas as pd
import pickle
from multiprocessing import Pool
from collections import defaultdict
import re

def read_one_file(path):
    res = ''
    with open(path) as f:
        res = f.read().strip()
    return res

def read_all_files(dir_path, processor=8):
    dataset = []
    files = glob.glob(dir_path + "/*.txt")

    if processor <= 1:
        for path in files:
            dataset.append(read_one_file(path))
    else:
        res = []
        pool = Pool(processor)
        for path in files:
            ret = pool.apply_async(read_one_file, args=(path,))
            res.append(ret)
        pool.close()
        pool.join()
        for v in res:
            data = v.get()
            dataset.append(data)
    return dataset

def gen_data(data_dir='./aclImdb'):
    num_proc = 8
    valid_frac = 0.2
    train_file = os.path.join(data_dir, 'dataset_train.pkl')
    valid_file = os.path.join(data_dir, 'dataset_valid.pkl')
    test_file = os.path.join(data_dir, 'dataset_test.pkl')

    train_neg = read_all_files(data_dir + '/train/neg', num_proc)
    train_pos = read_all_files(data_dir + '/train/pos', num_proc)
    test_neg = read_all_files(data_dir + '/test/neg', num_proc)
    test_pos = read_all_files(data_dir + '/test/pos', num_proc)

    train_dataset = pd.concat([pd.DataFrame({'review': train_pos, 'label':1}),
                         pd.DataFrame({'review': train_neg, 'label':0})],
                         axis=0, ignore_index=True)
    train_dataset.drop_duplicates(keep='first', inplace=True)

    valid_dataset = train_dataset.sample(frac=valid_frac, axis=0, random_state=2021)
    train_dataset.drop(valid_dataset.index, inplace=True)

    valid_dataset.reset_index(drop=True, inplace=True) # also: valid_dataset.index = pd.RangeIndex(len(valid_dataset))
    train_dataset.reset_index(drop=True, inplace=True)
    train_dataset.to_pickle(train_file)
    valid_dataset.to_pickle(valid_file)

    test_dataset = pd.concat([pd.DataFrame({'review': test_pos, 'label':1}),
                         pd.DataFrame({'review': test_neg, 'label':0})],
                         axis=0, ignore_index=True)
    test_dataset.to_pickle(test_file)
    print(len(train_dataset))
    print(len(valid_dataset))
    print(len(test_dataset))

def explore_data(path):
    print("----------------------\nexplore: {}".format(path))
    data = pd.read_pickle(path)
    print(data.info())
    print(data.describe())
    a = map(lambda x:len(x), data['review'])
    df = pd.DataFrame(a, columns=['length'])
    print(df.describe())

def normalize_string(s):
    s = re.sub(r"<br />",r" ",s)
    s = re.sub(r"(\d+\.\d+|[.!?,;:/\"\(\)\\])", r" \1 ", s)
    s = re.sub('[ ]+', ' ', s)
    return s

def build_vocab(data_file, out_file, least_df=3, need_lower=True):
    freq = defaultdict(int) # word freq
    df = defaultdict(int)   # document freq
    data = pd.read_pickle(data_file)
    for line in data['review']:
        if need_lower:
            line = line.lower()
        line = normalize_string(line)
        words = line.strip().split()
        words = [w.strip() for w in words]
        words = [w for w in words if len(w) >=2]
        for w in words:
            freq[w] += 1
        for w in set(words):
            df[w] += 1

    total_valid_words = {}
    for k,v in df.items():
        if v >= least_df:
            total_valid_words[k] = v

    total_valid_words = {w:(i+1) for i,w in enumerate(total_valid_words.keys())}
    total_valid_words['<pad>'] = 0

    with open(out_file, 'wb') as f:
        pickle.dump(total_valid_words, f)

    for t in sorted(df.items(), key=lambda x: -x[1]):
        word,index = t
        print("%s\t%d\t%d" % (word,index,freq[word]))
    print("word_vocab_size:{}".format(len(total_valid_words)))

if __name__ == '__main__':
    data_dir = '/data1/aclImdb'

    gen_data(data_dir)
    #explore_data(data_dir + "/dataset_train.pkl")
    #explore_data(data_dir + "/dataset_valid.pkl")
    #explore_data(data_dir + "/dataset_test.pkl")

    data_path = data_dir + "/dataset_train.pkl"
    out_file = '.conf/word_vocab.pkl'
    build_vocab(data_path, out_file)
