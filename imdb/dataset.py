import pandas
import numpy as np
import pickle
from torch.utils.data import Dataset
import re

class CharLevelDataset(Dataset):
    def __init__(self, pkl_file, max_len=4096, pad_idx=0):
        df = pandas.read_pickle(pkl_file)
        self.data = [np.array(self.padding_str(text, max_len, pad_idx)) for text in df['review']]
        self.labels = df['label'].to_numpy()
        self.vocab_size = 256

    def padding_str(self, sequences, max_len, pad, pos='back'):
        sequences = [ord(s) % 256 for s in sequences]
        if len(sequences) > max_len:
            sequences = sequences[:max_len]
        if pos == 'pre':
            sequences = [pad] * (max_len - len(sequences)) + sequences
        else:
            sequences = sequences + [pad] * (max_len - len(sequences))
        return sequences

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    def get_vocab_size(self):
        return self.vocab_size

class TextProcesser:
    def __init__(self, vocab_file, stopword_file=None, pad_idx=0, lower=True):
        # load word index
        self.word_index = {}
        with open(vocab_file, 'rb') as f:
            self.word_index = pickle.load(f)

        # load stopwords
        self.stopwords = set()
        if stopword_file:
            with open(stopword_file) as f:
                words = [line.split()[0].strip() for line in f]
                if lower:
                    words = [w.lower() for w in words]
                self.stopwords = set(words)

        self.pad_idx = pad_idx
        self.lower = lower

    def normalize_string(self, s):
        s = re.sub(r"<br />",r" ",s)
        s = re.sub(r"(\d+\.\d+|[.!?,;:/\"\(\)\\])", r" \1 ", s)
        s = re.sub('[ ]+', ' ', s)
        return s

    def process(self, text, max_len):
        sequences = []
        if self.lower:
            text = text.lower()
        text = self.normalize_string(text)
        tokens = text.split()
        #print(len(tokens))
        for word in tokens:
            word = word.strip()
            if word in self.word_index and word not in self.stopwords:
                sequences.append(self.word_index[word])
        if len(sequences) > max_len:
            sequences = sequences[:max_len]
        sequences = sequences + [self.pad_idx] * (max_len - len(sequences))
        return sequences

    def get_vocab_size(self):
        return len(self.word_index)

class WordLevelDataset(Dataset):
    def __init__(self, data_file, vocab_file, stopword_file, max_len=4096, pad_idx=0):
        df = pandas.read_pickle(data_file)
        self.tp = TextProcesser(vocab_file, stopword_file, pad_idx)
        self.data = [np.array(self.tp.process(text, max_len)) for text in df['review']]
        self.labels = df['label'].to_numpy()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    def get_vocab_size(self):
        return self.tp.get_vocab_size()

if __name__ == '__main__':
    path = '/data1/aclImdb/dataset_train.pkl'
    #dataset = CharLevelDataset(path)
    #for v in dataset[:10]:
    #    print(len(v))
    #    print(v[:64])

    vocab_file = './conf/word_vocab.pkl'
    stopword_file = './conf/stopwords.txt'
    dataset = WordLevelDataset(path, vocab_file, stopword_file)
    for v in dataset[:10]:
        print(len(v))
        print(v)
