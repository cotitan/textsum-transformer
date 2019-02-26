import os
import torch
import argparse
from Beam import Beam
import torch.nn.functional as F


def print_summaries(summaries, vocab, output_dir, pattern='%d.txt'):
    """
    param summaries: in shape (seq_len, batch)
    """
    i2w = {key: value for value, key in vocab.items()}

    for idx in range(len(summaries)):
        fout = open(os.path.join(output_dir, pattern % idx), "w")
        line = [i2w[tok] for tok in summaries[idx]\
                if tok != vocab["</s>"] and tok != vocab['<pad>']]
        fout.write(" ".join(line) + "\n")
        fout.close()


def greedy(model, x, tgt_vocab, max_trg_len=15):
    y = torch.ones(len(x), max_trg_len, dtype=torch.long).cuda() * tgt_vocab["<pad>"]
    y[:, 0] = tgt_vocab["<s>"]
    for i in range(max_trg_len - 1):
        logits = model(x, y)
        y[:, i + 1] = torch.argmax(logits[:, i, :], dim=-1)
    return y[:, 1:].detach().cpu().tolist()


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
            batch_x = valid_x.next_batch().cuda()
            if search == "greedy":
                summary = greedy(model, batch_x, tgt_vocab)
            elif search == "beam":
                summary = beam_search(model, batch_x, tgt_vocab, k=beam_width)
            else:
                raise NameError("Unknown search method")
            summaries.extend(summary)
    return summaries

