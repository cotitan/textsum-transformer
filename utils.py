import json
import numpy as np
from collections import defaultdict
import torch
import os
import config
import random

pad_tok = config.pad_tok
start_tok = config.start_tok
end_tok = config.end_tok
unk_tok = config.unk_tok

pad_index = config.pad_index

""" Caution:
In training data, unk_tok='<unk>', but in test data, unk_tok='UNK'.
This is reasonable, because if the unk_tok you predict is the same as the
unk_tok in the test data, then your prediction would be regard as correct,
but since unk_tok is unknown, it's impossible to give a correct prediction
"""


def my_pad_sequence(batch, pad_tok):
    max_len = max([len(b) for b in batch])
    batch = [b + [pad_tok] * (max_len - len(b)) for b in batch]
    return batch


def shuffle(bm1, bm2):
	c = list(zip(bm1.data, bm2.data))
	random.shuffle(c)
	bm1.data, bm2.data = zip(*c)
	return bm1, bm2


class BatchManager:
    def __init__(self, data, batch_size, vocab):
        self.steps = int(len(data) / batch_size)
        # comment following two lines to neglect the last batch
        if self.steps * batch_size < len(data):
            self.steps += 1
        self.vocab = vocab
        self.data = data
        self.batch_size = batch_size
        self.bid = 0

    def next_batch(self, pad_flag=True, cuda=True):
        stncs = list(self.data[self.bid * self.batch_size: (self.bid + 1) * self.batch_size])
        if pad_flag:
            stncs = my_pad_sequence(stncs, pad_tok)
            ids = [[self.vocab.get(tok, self.vocab[unk_tok]) for tok in stnc] for stnc in stncs]
            ids = torch.tensor(ids)
        self.bid += 1
        if self.bid == self.steps:
            self.bid = 0
        return stncs, ids.cuda() if cuda else ids


def build_vocab(filelist=['sumdata/train/train.article.txt', 'sumdata/train/train.title.txt'],
                vocab_file='sumdata/vocab.json', min_count=0, vocab_size=1e9):
    print("Building vocab with min_count=%d..." % min_count)
    word_freq = defaultdict(int)
    for file in filelist:
        fin = open(file, "r", encoding="utf8")
        for _, line in enumerate(fin):
            for word in line.strip().split():
                word_freq[word] += 1
        fin.close()
    print('Number of all words: %d' % len(word_freq))

    if unk_tok in word_freq:
        word_freq.pop(unk_tok)
    sorted_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    vocab = {pad_tok: 0, start_tok: 1, end_tok: 2, unk_tok: 3}
    for word, freq in sorted_freq:
        if freq > min_count:
            vocab[word] = len(vocab)
        if len(vocab) == vocab_size:
            break
    print('Number of filtered words: %d, %f%% ' % (len(vocab), len(vocab)/len(word_freq)*100))

    json.dump(vocab, open(vocab_file,'w'))
    return vocab

def get_vocab(TRAIN_X, TRAIN_Y):
    src_vocab_file = "sumdata/src_vocab.json"
    if not os.path.exists(src_vocab_file):
        src_vocab = build_vocab([TRAIN_X], src_vocab_file)
    else:
        src_vocab = json.load(open(src_vocab_file))

    tgt_vocab_file = "sumdata/tgt_vocab.json"
    if not os.path.exists(tgt_vocab_file):
        tgt_vocab = build_vocab([TRAIN_Y], tgt_vocab_file)
    else:
        tgt_vocab = json.load(open(tgt_vocab_file))
    return src_vocab, tgt_vocab


def load_data(filename, max_len, n_data=None):
    """
    :param filename: the file to read
    :param max_len: maximum length of a line
    :param vocab: dict {word: id}, if no vocab provided, return raw text
    :param n_data: number of lines to read
    :return: datas
    """
    fin = open(filename, "r", encoding="utf8")
    datas = []
    for idx, line in enumerate(fin):
        if idx == n_data or line == '':
            break
        words = line.strip().split()
        if len(words) > max_len - 2:
            words = words[:max_len-2]
        words = ['<s>'] + words + ['</s>']
        datas.append(words)
    return datas

