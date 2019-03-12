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
parser.add_argument('--ckpt_file', type=str, default='./models/elmo_small_2L_8H_512_epoch0.pkl', help='model file path')
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


def run_batch(batch_x, batch_y, model):
    x_stncs, x_ids = batch_x.next_batch()
    y_stncs, y_ids = batch_y.next_batch()

    logits = model(x_stncs, y_stncs, x_ids, y_ids)
    loss = model.loss_layer(logits.view(-1, logits.shape[-1]),
                            y_ids[:, 1:].contiguous().view(-1))
    return loss


def greedy(model, x_stncs, x_ids, tgt_vocab, max_trg_len=15):
    b_size = len(x_stncs)
    y_stncs = [['<s>'] * max_trg_len for _ in range(b_size)]
    y_ids = torch.ones(b_size, max_trg_len, dtype=torch.long).cuda()
    y_ids *= tgt_vocab['<s>']
    id2w = {v: k for k,v in tgt_vocab.items()}
    for i in range(max_trg_len - 1):
        logits = model.forward(x_stncs, y_stncs, x_ids, y_ids)
        argmax = torch.argmax(logits[:, i, :], dim=-1).detach().cpu().tolist()
        for j in range(b_size):
            y_stncs[j][i+1] = id2w[argmax[j]]
            y_ids[j][i+1] = argmax[j]
    return y_stncs


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
            x_stncs, x_ids = valid_x.next_batch()
            if search == "greedy":
                summary = greedy(model, x_stncs, x_ids, tgt_vocab)
            elif search == "beam":
                summary = beam_search(model, x_stncs, x_ids, tgt_vocab, k=beam_width)
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

    max_src_len = 60
    max_tgt_len = 20

    bs = args.batch_size
    n_test = args.n_test

    vocab = small_vocab
    test_x = BatchManager(load_data(TEST_X, max_src_len, n_test), bs, vocab)

    model = ElmoTransformer(max_src_len, len(vocab), 2, 8, 64, 64, 256, 512, 2048,
                            dropout=0.5, elmo_requires_grad=False).cuda()

    saved_state = torch.load(args.ckpt_file)
    model.load_state_dict(saved_state['state_dict'])
    print('Load model parameters from %s' % args.ckpt_file)

    my_test(test_x, model, small_vocab)


if __name__ == '__main__':
    main()

