import os
import json
import torch
import argparse
import numpy as np
from utils import BatchManager, load_data, get_vocab
from Transformer import Transformer
from Model import Model
from Beam import Beam
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Selective Encoding for Abstractive Sentence Summarization in pytorch')

parser.add_argument('--n_test', type=int, default=1951,
					help='Number of validation data (up to 189651 in gigaword) [default: 189651])')
parser.add_argument('--input_file', type=str, default="sumdata/Giga/input.txt", help='input file')
parser.add_argument('--output_dir', type=str, default="sumdata/Giga/systems/", help='')
parser.add_argument('--batch_size', type=int, default=64, help='Mini batch size [default: 32]')
parser.add_argument('--emb_dim', type=int, default=300, help='Embedding size [default: 256]')
parser.add_argument('--hid_dim', type=int, default=512, help='Hidden state size [default: 256]')
parser.add_argument('--maxout_dim', type=int, default=2, help='Maxout size [default: 2]')
parser.add_argument('--model_file', type=str, default='./models/params_0.pkl', help='model file path')
parser.add_argument('--search', type=str, default='greedy', help='greedy/beam')
parser.add_argument('--beam_width', type=int, default=12, help='beam search width')
args = parser.parse_args()
print(args)

if not os.path.exists(args.model_file):
	raise FileNotFoundError("model file not found")


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
	for i in range(max_trg_len-1):
		logits = model(x, y)
		y[:,i+1] = torch.argmax(logits[:,i,:])
	return y


def beam_search(model, batch_x, max_trg_len=10, k=args.beam_width):
	enc_outs, hidden = model.encode(batch_x)
	hidden = model.init_decoder_hidden(hidden)

	beams = [Beam(k, model.vocab, hidden[:,i,:])
			for i in range(batch_x.shape[0])]
	
	for _ in range(max_trg_len):
		for j in range(len(beams)):
			hidden = beams[j].get_hidden_state()
			word = beams[j].get_current_word()
			enc_outs_j = enc_outs[j].unsqueeze(0).expand(k, -1, -1)
			logit, hidden = model.decode(word, enc_outs_j, hidden)
			# logit: [k x V], hidden: [k x hid_dim]
			log_probs = F.softmax(logit, -1)
			beams[j].advance_(log_probs, hidden)

	allHyp, allScores = [], []
	n_best = 1
	for b in range(batch_x.shape[0]):
		scores, ks = beams[b].sort_best()
		allScores += [scores[:n_best]]
		hyps = [beams[b].get_hyp(k) for k in ks[:n_best]]
		allHyp.append(hyps)

	# shape of allHyp: [batch, 1, list]
	allHyp = [[int(w.cpu().numpy()) for w in hyp[0]] for hyp in allHyp]
	return allHyp


def my_test(valid_x, model, tgt_vocab):
	summaries = []
	with torch.no_grad():
		for _ in range(valid_x.steps):
			batch_x = valid_x.next_batch().cuda()
			if args.search == "greedy":
				summary = greedy(model, batch_x, tgt_vocab)
			elif args.search == "beam":
				summary = beam_search(model, batch_x)
			else:
				raise NameError("Unknown search method")
			summaries.extend(summary)
	print_summaries(summaries, tgt_vocab)
	print("Done!")


def main():
	data_dir = '/home/tiankeke/workspace/datas/sumdata/'
	TRAIN_X = os.path.join(data_dir, 'train/train.article.txt')
	TRAIN_Y = os.path.join(data_dir, 'train/train.title.txt')
	TEST_X = os.path.join(data_dir, "Giga/input.txt")
	
	src_vocab, tgt_vocab = get_vocab(TRAIN_X, TRAIN_Y)
	max_src_len = 101
	max_tgt_len = 47
	
	test_x = BatchManager(load_data(TEST_X, src_vocab, max_src_len, args.n_test), args.batch_size*4)
	
	model = Transformer(len(src_vocab), len(tgt_vocab), max_src_len, max_tgt_len,
			d_word_vec=300, N=6, n_head=3, d_q=100, d_k=100, d_v=100, d_model=300, d_inner=600,
			dropout=0.1, tgt_emb_prj_weight_share=True).cuda()
	print(model)
	# model.eval()
	for _ in range(test_x.steps):
		x = test_x.next_batch()
		print(x.shape)

	file = args.model_file
	if os.path.exists(file):
		model.load_state_dict(torch.load(file))
		print('Load model parameters from %s' % file)

	my_test(test_x, model, tgt_vocab)


if __name__ == '__main__':
	main()

