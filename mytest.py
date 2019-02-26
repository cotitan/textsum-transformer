import os
import json
import torch
import argparse
import numpy as np
from utils import BatchManager, load_data, get_vocab, build_vocab
from Transformer import Transformer, TransformerShareEmbedding
from translate import translate, print_summaries

parser = argparse.ArgumentParser(description='Selective Encoding for Abstractive Sentence Summarization in pytorch')

parser.add_argument('--n_test', type=int, default=1951,
                    help='Number of validation data (up to 189651 in gigaword) [default: 189651])')
parser.add_argument('--input_file', type=str, default="sumdata/Giga/input.txt", help='input file')
parser.add_argument('--output_dir', type=str, default="sumdata/Giga/systems/", help='')
parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size [default: 32]')
parser.add_argument('--ckpt_file', type=str, default='./models/params_v2_0.pkl', help='model file path')
parser.add_argument('--search', type=str, default='greedy', help='greedy/beam')
parser.add_argument('--beam_width', type=int, default=12, help='beam search width')
args = parser.parse_args()
print(args)


def my_test(valid_x, model, tgt_vocab):
    summaries = translate(valid_x, model, tgt_vocab, search='greedy')
    print_summaries(summaries, tgt_vocab, args.output_dir)
    print("Done!")


def main():
    if not os.path.exists(args.ckpt_file):
        raise FileNotFoundError("model file not found")

    data_dir = '/home/tiankeke/workspace/datas/sumdata/'
    TRAIN_X = os.path.join(data_dir, 'train/train.article.txt')
    TRAIN_Y = os.path.join(data_dir, 'train/train.title.txt')
    TEST_X = args.input_file

    small_vocab_file = 'sumdata/small_vocab.json'
    if os.path.exists(small_vocab_file):
        small_vocab = json.load(open(small_vocab_file))
    else:
        small_vocab = build_vocab([TRAIN_X, TRAIN_Y], small_vocab_file, vocab_size=80000)

    max_src_len = 101
    max_tgt_len = 47

    test_x = BatchManager(load_data(TEST_X, max_src_len, args.n_test, small_vocab), args.batch_size)

    model = TransformerShareEmbedding(len(small_vocab), max_src_len, 1, 6,
                                      300, 50, 50, 1200, False).cuda()

    saved_state = torch.load(args.ckpt_file)
    model.load_state_dict(saved_state['state_dict'])
    print('Load model parameters from %s' % args.ckpt_file)

    my_test(test_x, model, small_vocab)


if __name__ == '__main__':
    main()

