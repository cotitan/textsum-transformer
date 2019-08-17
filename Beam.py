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

        self.length=1
        self.lengths = self.tt.FloatTensor(size, device=self.device).fill_(12)
        # max_trg_len+1, include eos
        self.sequence = self.tt.LongTensor(size, max_trg_len+1, device=self.device).fill_(self.pad)
        self.sequence[0, 0] = self.bos

    def get_sequence(self):
        return self.sequence

    def advance(self, log_probs):
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

    def advance_v1(self, log_probs):
        if self.done:
            return True

        """Advance the beam."""
        log_probs = log_probs.squeeze() # k*V
        num_words = log_probs.shape[-1]

        non_eos_mask = torch.ones_like(log_probs).cuda()
        non_eos_mask[:, self.eos] = 0
        # for i in range(self.size):
        #     if self.sequence[i][int(self.lengths[i])-1] == self.eos:
        #         non_eos_mask[i,:] = 0

        # Sum the previous scores.
        if self.length > 1:
            scores = log_probs + self.scores.unsqueeze(1).expand_as(log_probs)
            new_len = self.lengths.unsqueeze(1).expand_as(scores) + non_eos_mask
            normalized_scores = scores / new_len
        else:
            scores = log_probs[0]
            normalized_scores = scores

        flat_scores = normalized_scores.view(-1)

        bestScores, bestScoresId = flat_scores.topk(self.size, 0, True, True)
        self.scores = scores.view(-1)[bestScoresId]

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId / num_words
        word_idx = bestScoresId % num_words

        self.sequence = self.sequence[prev_k]
        self.sequence[:, self.length] = word_idx  # = bestScoresId - prev_k * num_words

        self.length += 1
        self.lengths[word_idx!=self.eos] += 1

        # End condition is when top-of-beam is EOS.
        if self.sequence[0,-1] == self.eos:
            self.done = True

    def get_hyp(self):
        return self.sequence[0]


# import torch


# class Beam(object):
#     """Ordered beam of candidate outputs."""

#     def __init__(self, size, vocab, max_trg_len, device=torch.device("cuda:0"), use_ptr_gen=False):
#         """Initialize params."""
#         self.size = size
#         self.done = False
#         self.vocab = vocab
#         self.rev_vocab = {v:k for k,v in vocab.items()}
#         self.use_ptr_gen = True
#         # self.pad = vocab['<pad>']
#         self.bos = vocab['<s>']
#         self.eos = vocab['</s>']
#         self.pad = vocab['<pad>']
#         self.device = device
#         self.tt = torch.cuda if device.type == "cuda" else torch
#         # The score for each translation on the beam.
#         self.scores = self.tt.FloatTensor(size, device=self.device).zero_()

#         self.length=1
#         # fill_(12) lead to better performance on giga, 0.1 for DUC2004
#         self.lengths = self.tt.FloatTensor(size, device=self.device).fill_(18) 
#         # max_trg_len+1, include eos
#         self.ys = self.tt.LongTensor(size, max_trg_len+1, device=self.device).fill_(self.pad)
#         self.ys[0, 0] = self.bos
#         self.ext_ys = self.tt.LongTensor(size, max_trg_len+1, device=self.device).fill_(self.pad)
#         self.ext_ys[0,0] = self.bos

#     def get_sequence(self):
#         return self.ys

#     def advance(self, log_probs):
#         if self.done:
#             return True

#         """Advance the beam."""
#         log_probs = log_probs.squeeze() # k*V
#         num_words = log_probs.shape[-1]

#         # Sum the previous scores.
#         if self.length > 1:
#             scores = log_probs + self.scores.unsqueeze(1).expand_as(log_probs)
#         else:
#             scores = log_probs[0]

#         flat_scores = scores.view(-1)

#         bestScores, bestScoresId = flat_scores.topk(self.size, 0, True, True)
#         self.scores = bestScores

#         # bestScoresId is flattened beam x word array, so calculate which
#         # word and beam each score came from
#         prev_k = bestScoresId / num_words
#         word_idx = bestScoresId % num_words

#         self.ext_ys = self.ext_ys[prev_k]
#         self.ext_ys[:, self.length] = word_idx  # = bestScoresId - prev_k * num_words
#         self.ys = self.ys[prev_k]
#         for j in range(len(self.ys)):
#             self.ys[j,self.length] = word_idx[j] if int(word_idx[j].cpu().detach()) in self.rev_vocab else vocab['<unk>']

#         self.length += 1

#         # End condition is when top-of-beam is EOS.
#         if self.ys[0,-1] == self.eos:
#             self.done = True

#     def advance_v1(self, log_probs):
#         if self.done:
#             return True

#         """Advance the beam."""
#         log_probs = log_probs.squeeze() # k*V
#         num_words = log_probs.shape[-1]

#         non_eos_mask = torch.ones_like(log_probs).cuda()
#         non_eos_mask[:, self.eos] = 0
#         # for i in range(self.size):
#         #     if self.ys[i][int(self.lengths[i])-1] == self.eos:
#         #         non_eos_mask[i,:] = 0

#         # Sum the previous scores.
#         if self.length > 1:
#             scores = log_probs + self.scores.unsqueeze(1).expand_as(log_probs)
#             new_len = self.lengths.unsqueeze(1).expand_as(scores) + non_eos_mask
#             normalized_scores = scores / new_len
#         else:
#             scores = log_probs[0]
#             normalized_scores = scores

#         flat_scores = normalized_scores.view(-1)

#         bestScores, bestScoresId = flat_scores.topk(self.size, 0, True, True)
#         self.scores = scores.view(-1)[bestScoresId]

#         # bestScoresId is flattened beam x word array, so calculate which
#         # word and beam each score came from
#         prev_k = bestScoresId / num_words
#         word_idx = bestScoresId % num_words

#         self.ext_ys = self.ext_ys[prev_k]
#         self.ext_ys[:, self.length] = word_idx  # = bestScoresId - prev_k * num_words

#         self.ys = self.ys[prev_k]
#         for j in range(len(self.ys)):
#             self.ys[j, self.length] = word_idx[j] if int(word_idx[j].cpu().detach()) in self.rev_vocab else vocab['<unk>']
#         # self.ys[:, self.length] = word_idx  # = bestScoresId - prev_k * num_words

#         self.length += 1
#         self.lengths = self.lengths[prev_k]
#         self.lengths[word_idx!=self.eos] += 1

#         # End condition is when top-of-beam is EOS.
#         if self.ys[0,-1] == self.eos:
#             self.done = True

#     def get_hyp(self):
#         return self.ext_ys[0] if self.use_ptr_gen else self.ys[0]
