import os
import json
import torch
import argparse
from utils import BatchManager, load_data, get_vocab, build_vocab
from transformer.Models import Transformer
from Beam import Beam
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Selective Encoding for Abstractive Sentence Summarization in pytorch')

parser.add_argument('--n_test', type=int, default=1951,
                    help='Number of validation data (up to 189651 in gigaword) [default: 189651])')
parser.add_argument('--input_file', type=str, default="sumdata/Giga/input.txt", help='input file')
parser.add_argument('--output_dir', type=str, default="sumdata/Giga/systems/", help='')
parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size [default: 32]')
parser.add_argument('--emb_dim', type=int, default=300, help='Embedding size [default: 256]')
parser.add_argument('--hid_dim', type=int, default=512, help='Hidden state size [default: 256]')
parser.add_argument('--maxout_dim', type=int, default=2, help='Maxout size [default: 2]')
parser.add_argument('--ckpt_file', type=str, default='./models/params_v1_0.pkl', help='model file path')
parser.add_argument('--search', type=str, default='greedy', help='greedy/beam')
parser.add_argument('--beam_width', type=int, default=12, help='beam search width')
args = parser.parse_args()
print(args)


def print_summaries(summaries, vocab):
    """
    param summaries: in shape (seq_len, batch)
    """
    i2w = {key: value for value, key in vocab.items()}

    for idx in range(len(summaries)):
        fout = open(os.path.join(args.output_dir, "%d.txt" % idx), "w")
        line = [i2w[tok] for tok in summaries[idx] if tok != vocab["</s>"]]
        fout.write(" ".join(line) + "\n")
        fout.close()


def greedy(model, x, tgt_vocab, max_trg_len=15):
    y = torch.ones(x.shape[0], max_trg_len, dtype=torch.long).cuda() * tgt_vocab["<pad>"]
    y[:,0] = tgt_vocab["<s>"]

    pos_x = torch.arange(x.shape[1]).unsqueeze(0).expand_as(x).cuda()
    pos_y = torch.arange(y.shape[1]).unsqueeze(0).expand_as(y).cuda()

    for i in range(max_trg_len-1):
        logits = model(x, pos_x, y, pos_y)
        y[:, i+1] = torch.argmax(logits[:,i,:], dim=-1)
    return y[:,1:].detach().cpu().tolist()


def beam_search(model, batch_x, vocab, max_trg_len=10, k=args.beam_width):

    beams = [Beam(k, vocab, max_trg_len) for _ in range(batch_x.shape[0])]

    for i in range(max_trg_len):
        for j in range(len(beams)):
            x = batch_x[j].unsqueeze(0).expand(k, -1)
            y = beams[j].get_sequence()

            pos_x = torch.arange(x.shape[1]).unsqueeze(0).expand_as(x).cuda()
            pos_y = torch.arange(y.shape[1]).unsqueeze(0).expand_as(y).cuda()

            logit = model(x, pos_x, y, pos_y)
            # logit: [k, seqlen, V]
            log_probs = torch.log(F.softmax(logit[:, i, :], -1))
            beams[j].advance_(log_probs)

    allHyp = [b.get_hyp().cpu().numpy() for b in beams]
    return allHyp


def my_test(valid_x, model, tgt_vocab):
    summaries = []
    with torch.no_grad():
        for i in range(valid_x.steps):
            _, x = valid_x.next_batch()
            if args.search == "greedy":
                summary = greedy(model, x, tgt_vocab)
            elif args.search == "beam":
                summary = beam_search(model, x, tgt_vocab)
            else:
                raise NameError("Unknown search method")
            summaries.extend(summary)
    print_summaries(summaries, tgt_vocab)
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

    test_x = BatchManager(load_data(TEST_X, max_src_len, args.n_test), args.batch_size, small_vocab)

    model = Transformer(len(small_vocab), len(small_vocab), max_src_len, d_word_vec=300,
                        d_model=300, d_inner=1200, n_layers=1, n_head=6, d_k=50,
                        d_v=50, dropout=0.1, tgt_emb_prj_weight_sharing=True,
                        emb_src_tgt_weight_sharing=True).cuda()
    # print(model)
    model.eval()

    saved_state = torch.load(args.ckpt_file)
    model.load_state_dict(saved_state['state_dict'])
    print('Load model parameters from %s' % args.ckpt_file)

    my_test(test_x, model, small_vocab)


if __name__ == '__main__':
    main()

