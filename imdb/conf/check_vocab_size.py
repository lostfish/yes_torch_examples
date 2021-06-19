
import pickle
import sys

vocab_file = 'word_vocab.pkl'
with open(vocab_file, 'rb') as f:
    vocabs = pickle.load(f)
    print(len(vocabs))
