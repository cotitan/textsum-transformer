import os
import json
import torch
import argparse
from utils import BatchManager, load_data, get_vocab, build_vocab
from Transformer import ElmoTransformer
from Beam import Beam
import torch.nn.functional as F
import copy

parser = argparse.ArgumentParser(description='Selective Encoding for Abstractive Sentence Summarization in pytorch')

parser.add_argument('--n_test', type=int, default=1951,
                    help='Number of validation data (up to 189651 in gigaword) [default: 189651])')
parser.add_argument('--input_file', type=str, default="sumdata/Giga/input.txt", help='input file')
parser.add_argument('--output_dir', type=str, default="sumdata/Giga/systems/", help='')
parser.add_argument('--batch_size', type=int, default=64, help='Mini batch size [default: 32]')
parser.add_argument('--ckpt_file', type=str, default='./models/params_elmo_0.pkl', help='model file path')
parser.add_argument('--search', type=str, default='greedy', help='greedy/beam')
parser.add_argument('--beam_width', type=int, default=12, help='beam search width')
args = parser.parse_args()
print(args)


def print_summaries(summaries, vocab, output_dir, pattern='%d.txt'):
    """
    param summaries: in shape (seq_len, batch)
    """
    for idx in range(len(summaries)):
        fout = open(os.path.join(output_dir, pattern % idx), "w")
        line = [tok for tok in summaries[idx] if tok not in ['<s>', '</s>', '<pad>']]
        fout.write(" ".join(line) + "\n")
        fout.close()


def greedy(model, x, tgt_vocab, max_trg_len=15):
    y = [['<s>'] * max_trg_len for _ in range(len(x))]
    id2w = {v: k for k,v in tgt_vocab.items()}
    for i in range(max_trg_len - 1):
        logits = model(x, y)
        argmax = torch.argmax(logits[:, i, :], dim=-1).detach().cpu().tolist()
        for j in range(len(y)):
            y[j][i+1] = id2w[argmax[j]]
    return y


def beam_search(model, batch_x, vocab, max_trg_len=10, k=3):
    beams = [Beam(k, vocab, max_trg_len) for _ in range(batch_x.shape[0])]

    for i in range(max_trg_len):
        for j in range(len(beams)):
            x = batch_x[j].unsqueeze(0).expand(k, -1)
            y = beams[j].get_sequence()
            logit = model(x, y)
            # logit: [k, seqlen, V]
            log_probs = torch.log(F.softmax(logit[:, i, :], -1))
            beams[j].advance_(log_probs)

    allHyp = [b.get_hyp().cpu().numpy() for b in beams]
    return allHyp


def translate(valid_x, model, tgt_vocab, search='greedy', beam_width=5):
    summaries = []
    model.eval()
    with torch.no_grad():
        for i in range(valid_x.steps):
            print(i, flush=True)
            batch_x = valid_x.next_batch()
            if search == "greedy":
                summary = greedy(model, batch_x, tgt_vocab)
            elif search == "beam":
                summary = beam_search(model, batch_x, tgt_vocab, k=beam_width)
            else:
                raise NameError("Unknown search method")
            summaries.extend(summary)
    return summaries


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

    vocab = small_vocab
    test_x = BatchManager(load_data(TEST_X, max_src_len, args.n_test), args.batch_size, pad=False)

    model = ElmoTransformer(max_src_len, len(vocab), 1, 4, 64, 64, 256, 1024).cuda()

    saved_state = torch.load(args.ckpt_file)
    model.load_state_dict(saved_state['state_dict'])
    print('Load model parameters from %s' % args.ckpt_file)

    my_test(test_x, model, small_vocab)


if __name__ == '__main__':
    main()

