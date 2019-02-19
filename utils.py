import json
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import threading
import os

start_tok = "<s>"
end_tok = "</s>"
unk_tok = "<unk>"
pad_tok = "<pad>"
""" Caution:
In training data, unk_tok='<unk>', but in test data, unk_tok='UNK'.
This is reasonable, because if the unk_tok you predict is the same as the
unk_tok in the test data, then your prediction would be regard as correct,
but since unk_tok is unknown, it's impossible to give a correct prediction
"""


def my_pad_sequence(batch, pad_value):
    max_len = max([len(b) for b in batch])
    batch = [b + [pad_value] * (max_len - len(b)) for b in batch]
    return torch.tensor(batch)


class BatchManager:
    def __init__(self, data, batch_size):
        self.steps = int(len(data) / batch_size)
        # comment following two lines to neglect the last batch
        if self.steps * batch_size < len(data):
            self.steps += 1
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
            # batch = collate_fn(batch, pad_value=3)
            batch = my_pad_sequence(batch, 3)
            self.bid += 1
            if self.bid == self.steps:
                self.bid = 0
            self.buffer.append(batch)

    def next_batch(self):
        batch = list(self.data[self.bid * self.batch_size: (self.bid + 1) * self.batch_size])
        batch = my_pad_sequence(batch, 3)
        self.bid += 1
        if self.bid == self.steps:
            self.bid = 0

        """ for multi-thread reading """
        # batch = self.buffer.pop()
        # self.s1.release()
        return batch


class myCollate:
    def __init__(self, pad_value=3):
        self.pad_value = pad_value
        
    def collate_fn(self, batch_data):
        batch_data.sort(key=lambda x: len(x), reverse=True)
        batch_data = [torch.tensor(x) for x in batch_data]
        padded = pad_sequence(batch_data, batch_first=True, padding_value=self.pad_value)
        # packed = pack_padded_sequence(padded, lens, batch_first=True)
        return padded
    
    def __call__(self, batch_data):
        return self.collate_fn(batch_data)


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

    vocab = {start_tok: 0, end_tok: 1, unk_tok: 2, pad_tok: 3}
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


def build_vocab_from_embeddings(embedding_path, data_file_list):
    embedding_vocab = load_embedding_vocab(embedding_path)
    vocab = {start_tok: 0, end_tok: 1, unk_tok: 2, pad_tok: 3}

    for file in data_file_list:
        fin = open(file)
        for _, line in enumerate(fin):
            for word in line.split():
                if (word in embedding_vocab) and (word not in vocab):
                    vocab[word] = len(vocab)
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


def load_data(filename, vocab, max_len, n_data=None, target=False):
    fin = open(filename, "r", encoding="utf8")
    datas = []
    for idx, line in enumerate(fin):
        if idx == n_data or line == '':
            break
        words = line.strip().split()
        if len(words) > max_len - 2:
            words = words[:max_len-2]
        words = ['<s>'] + words + ['</s>']
        sample = [vocab.get(w, vocab[unk_tok]) for w in words]
        datas.append(sample)
    return datas


class MyDatasets(Dataset):
    def __init__(self, filename, vocab, n_data=None):
        self.datas = load_data(filename, vocab, n_data)
        self._size = len(self.datas)
    
    def __getitem__(self, idx):
        return self.datas[idx]
    
    def __len__(self):
        return self._size


def getDataLoader(filepath, vocab, n_data, batch_size, num_workers=0):
    dataset = MyDatasets(filepath, vocab, n_data)
    loader = DataLoader(dataset, batch_size, num_workers=num_workers, collate_fn=myCollate(vocab[pad_tok]))
    return loader
