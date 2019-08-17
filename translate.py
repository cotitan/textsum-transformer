import os
import torch
import argparse
from Beam import Beam
import torch.nn.functional as F
import time


def print_summaries(summaries, vocab, output_dir, pattern='%d.txt'):
    """
    param summaries: in shape (seq_len, batch)
    """
    i2w = {key: value for value, key in vocab.items()}
    i2w[vocab['<unk>']] = 'UNK'

    for idx in range(len(summaries)):
        fout = open(os.path.join(output_dir, pattern % idx), "w")
        line = [summaries[idx][0]]
        for tok in summaries[idx][1:]:
            if tok in [vocab['</s>'], vocab['<pad>']]:
                break
            if tok != line[-1]:
                line.append(tok)
        if len(line)==0:
            line.append(3) # 3 for unk
        line = [i2w[tok] for tok in line]
        fout.write(" ".join(line[1:]) + "\n")
        fout.close()


def greedy(model, x, tgt_vocab, max_trg_len=20, repl_unk=False):
    y = torch.ones(len(x), max_trg_len, dtype=torch.long).cuda() * tgt_vocab["<pad>"]
    y[:, 0] = tgt_vocab["<s>"]
    enc_outputs = model.encode(x)
    # print(enc_outputs.shape)
    for i in range(max_trg_len - 1):
        logits, dec_enc_attns = model.decode(enc_outputs, x, y[:, :i+1])
        y[:, i + 1] = torch.argmax(logits[:, i, :], dim=-1)
        if repl_unk:
            argmax = dec_enc_attns[:,i,:].argmax(dim=-1)
            for j in range(y.shape[0]):
                if int(y[j,i+1].cpu().detach()) == tgt_vocab['<unk>']:
                    y[j,i+1] == x[j, int(argmax[j].cpu().detach())]
    return y.detach().cpu().tolist(), dec_enc_attns


def beam_search(model, batch_x, vocab, max_trg_len=18, k=3):
    beams = [Beam(k, vocab, max_trg_len) for _ in range(batch_x.shape[0])]
    enc_outputs = model.encode(batch_x)

    for i in range(max_trg_len):
        todo = [j for j in range(len(beams)) if not beams[j].done]
        xs = torch.cat([batch_x[j].unsqueeze(0).expand(k, -1) for j in todo], dim=0)
        ys = torch.cat([beams[j].get_sequence() for j in todo], dim=0)
        enc_outs = torch.cat([enc_outputs[j].unsqueeze(0).expand(k, -1, -1) for j in todo], dim=0)
        logits, *_ = model.decode(enc_outs, xs, ys[:, :i+1])
        log_probs = torch.log(F.softmax(logits[:, i, :], -1))
        idx = 0
        for j in todo:
            beams[j].advance_v1(log_probs[idx: idx+k])
            idx += k

    allHyp = [b.get_hyp().cpu().numpy() for b in beams]
    return allHyp


def translate(valid_x, model, tgt_vocab, search='greedy', beam_width=5):
    summaries = []
    model.eval()
    start = time.time()
    with torch.no_grad():
        for i in range(valid_x.steps):
            _, batch_x = valid_x.next_batch()
            if search == "greedy":
                summary, dec_enc_attns = greedy(model, batch_x, tgt_vocab)
            elif search == "beam":
                summary = beam_search(model, batch_x, tgt_vocab, k=beam_width)
            else:
                raise NameError("Unknown search method")
            summaries.extend(summary)
    end = time.time()
    print('%.1f seconds spent, speed=%f/seconds' % (end-start, len(valid_x.data)/(end-start)))
    return summaries

