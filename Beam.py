"""Beam search implementation in PyTorch."""
#
#
#         hyp1#-hyp1---hyp1 -hyp1
#                 \             /
#         hyp2 \-hyp2 /-hyp2#hyp2
#                               /      \
#         hyp3#-hyp3---hyp3 -hyp3
#         ========================
#
# Takes care of beams, back pointers, and scores.

# Code borrowed from PyTorch OpenNMT example
# https://github.com/pytorch/examples/blob/master/OpenNMT/onmt/Beam.py

import torch


class Beam(object):
    """Ordered beam of candidate outputs."""

    def __init__(self, size, vocab, max_trg_len, device=torch.device("cuda:0")):
        """Initialize params."""
        self.size = size
        self.done = False
        # self.pad = vocab['<pad>']
        self.bos = vocab['<s>']
        self.eos = vocab['</s>']
        self.pad = vocab['<pad>']
        self.device = device
        self.tt = torch.cuda if device.type == "cuda" else torch
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size, device=self.device).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size, device=self.device).fill_(self.eos)]
        self.nextYs[0][0] = self.bos

        self.length=1
        # max_trg_len+1, include eos
        self.sequence = self.tt.LongTensor(size, max_trg_len+1, device=self.device).fill_(self.pad)
        self.sequence[:, 0] = self.eos

    def get_sequence(self):
        return self.sequence

    def advance_(self, log_probs):
        if self.done:
            return True

        """Advance the beam."""
        log_probs = log_probs.squeeze() # k*V
        num_words = log_probs.shape[-1]

        # Sum the previous scores.
        if self.length > 1:
            scores = log_probs + self.scores.unsqueeze(1).expand_as(log_probs)
        else:
            scores = log_probs[0]

        flat_scores = scores.view(-1)

        bestScores, bestScoresId = flat_scores.topk(self.size, 0, True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId / num_words
        word_idx = bestScoresId % num_words

        self.sequence = self.sequence[prev_k]
        self.sequence[:, self.length] = word_idx  # = bestScoresId - prev_k * num_words

        self.length += 1

        # End condition is when top-of-beam is EOS.
        if self.sequence[0,-1] == self.eos:
            self.done = True

    def get_hyp(self):
        return self.sequence[0]
