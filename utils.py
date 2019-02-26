import json
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import threading
import word2vec
import os
import config

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


def my_pad_sequence(batch, pad_index):
    max_len = max([len(b) for b in batch])
    batch = [b + [pad_index] * (max_len - len(b)) for b in batch]
    return torch.tensor(batch)


class BatchManager:
    def __init__(self, data, batch_size, pad=True):
        self.steps = int(len(data) / batch_size)
        # comment following two lines to neglect the last batch
        if self.steps * batch_size < len(data):
            self.steps += 1
        self.pad = pad
        self.data = data
        self.batch_size = batch_size
        self.bid = 0

        """ for multi-thread reading """
        # self.buffer = []
        # self.s1 = threading.Semaphore(1)
        # self.t1 = threading.Thread(target=self.loader, args=())
        # self.t1.start()

    def loader(self):
        while True:
            # generate next batch only when buffer is empty()
            self.s1.acquire()
            batch = list(self.data[self.bid * self.batch_size: (self.bid + 1) * self.batch_size])
            # batch = collate_fn(batch, pad_index)
            batch = my_pad_sequence(batch, pad_index)
            self.bid += 1
            if self.bid == self.steps:
                self.bid = 0
            self.buffer.append(batch)

    def next_batch(self):
        batch = list(self.data[self.bid * self.batch_size: (self.bid + 1) * self.batch_size])
        if self.pad:
            batch = my_pad_sequence(batch, pad_index)
        self.bid += 1
        if self.bid == self.steps:
            self.bid = 0

        """ for multi-thread reading """
        # batch = self.buffer.pop()
        # self.s1.release()
        return batch


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


def load_embedding_vocab(embedding_path):
    fin = open(embedding_path)
    vocab = set([])
    for _, line in enumerate(fin):
        vocab.add(line.split()[0])
    return vocab


def load_word2vec_embedding(filepath):
    w2v = word2vec.load(filepath)
    weights = w2v.vectors
    vocab = {}
    if pad_tok not in w2v.vocab:
        vocab[pad_tok] = 0
        weights = np.concatenate([np.zeros((1, weights.shape[1])), weights], axis=0)
    for tok in w2v.vocab:
        vocab[tok] = len(vocab)
    return vocab, torch.tensor(weights, dtype=torch.float)


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


def load_data(filename, max_len, n_data=None, vocab=None):
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
        if vocab is not None:
            words = [vocab.get(w, vocab[unk_tok]) for w in words]
        datas.append(words)
    return datas


class MyDatasets(Dataset):
    def __init__(self, filename, vocab, n_data=None):
        self.datas = load_data(filename, vocab, n_data)
        self._size = len(self.datas)
    
    def __getitem__(self, idx):
        return self.datas[idx]
    
    def __len__(self):
        return self._size


def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert(isinstance(size, torch.Size))
    return " × ".join(map(str, size))


def dump_tensors(gpu_only=True):
    """ GPU memory debugger
    Prints a list of the Tensors being tracked by the garbage collector."""
    import gc
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print("%s:%s%s %s" % (type(obj).__name__,
                                          " GPU" if obj.is_cuda else "",
                                          " pinned" if obj.is_pinned else "",
                                          pretty_size(obj.size())))
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print("%s → %s:%s%s%s%s %s" % (type(obj).__name__,
                                                   type(obj.data).__name__,
                                                   " GPU" if obj.is_cuda else "",
                                                   " pinned" if obj.data.is_pinned else "",
                                                   " grad" if obj.requires_grad else "",
                                                   " volatile" if obj.volatile else "",
                                                   pretty_size(obj.data.size())))
                    total_size += obj.data.numel()
        except Exception as e:
            pass
    print("Total size:", total_size)
