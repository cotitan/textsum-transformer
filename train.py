import os
import logging
import json
import utils
import torch
import argparse
from Model import Model
from Transformer import Transformer
from utils import BatchManager, load_data, get_vocab

parser = argparse.ArgumentParser(description='Selective Encoding for Abstractive Sentence Summarization in DyNet')

parser.add_argument('--n_epochs', type=int, default=5, help='Number of epochs [default: 3]')
parser.add_argument('--n_train', type=int, default=3803900,
					help='Number of training data (up to 3803957 in gigaword) [default: 3803957]')
parser.add_argument('--n_valid', type=int, default=189651,
					help='Number of validation data (up to 189651 in gigaword) [default: 189651])')
parser.add_argument('--batch_size', type=int, default=64, help='Mini batch size [default: 32]')
parser.add_argument('--emb_dim', type=int, default=300, help='Embedding size [default: 256]')
parser.add_argument('--hid_dim', type=int, default=512, help='Hidden state size [default: 256]')
parser.add_argument('--maxout_dim', type=int, default=2, help='Maxout size [default: 2]')
parser.add_argument('--model_file', type=str, default='./models/params_0.pkl')
args = parser.parse_args()


logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s',
	filename='log/train.log',
	filemode='w'
)

# define a new Handler to log to console as well
console = logging.StreamHandler()
# optional, set the logging level
console.setLevel(logging.INFO)
# set a format which is the same for console use
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


model_dir = './models'
if not os.path.exists(model_dir):
	os.mkdir(model_dir)


def run_batch(valid_x, valid_y, model):
	x = valid_x.next_batch().cuda()
	y = valid_y.next_batch().cuda()

	logits = model(x, y)

	loss = 0
	for i in range(y.shape[0]):
		loss += model.loss_layer(logits[i], y[i,1:])
	loss /= y.shape[0] # y.shape[1] == out_seq_len
	return loss


def train(train_x, train_y, valid_x, valid_y, model, optimizer, scheduler, epochs=1):
	logging.info("Start to train...")
	n_batches = train_x.steps
	model.train()
	for epoch in range(epochs):
		for idx in range(n_batches):
			optimizer.zero_grad()

			loss = run_batch(train_x, train_y, model)
			loss.backward()  # do not use retain_graph=True
			print(model.decoder.embedding.word_embeddings.weight.grad)
			# torch.nn.utils.clip_grad_value_(model.parameters(), 5)

			optimizer.step()
			# scheduler.step()

			if (idx + 1) % 50 == 0:
				train_loss = loss.cpu().detach().numpy()
				with torch.no_grad():
					valid_loss = run_batch(valid_x, valid_y, model)
				logging.info('epoch %d, step %d, training loss = %f, validation loss = %f'
							 % (epoch, idx + 1, train_loss, valid_loss))

		model.cpu()
		torch.save(model.state_dict(), os.path.join(model_dir, 'params_%d.pkl' % epoch))
		logging.info('Model saved in dir %s' % model_dir)
		model.cuda()
		# model.embedding_look_up.to(torch.device("cpu"))


def main():
	print(args)
	
	data_dir = '/home/tiankeke/workspace/datas/sumdata/'
	TRAIN_X = os.path.join(data_dir, 'train/train.article.txt')
	TRAIN_Y = os.path.join(data_dir, 'train/train.title.txt')
	VALID_X = os.path.join(data_dir, 'train/valid.article.filter.txt')
	VALID_Y = os.path.join(data_dir, 'train/valid.title.filter.txt')
	
	src_vocab, tgt_vocab = get_vocab(TRAIN_X, TRAIN_Y)
	max_src_len = 101
	max_tgt_len = 47
	
	train_x = BatchManager(load_data(TRAIN_X, src_vocab, max_src_len, args.n_train), args.batch_size)
	train_y = BatchManager(load_data(TRAIN_Y, tgt_vocab, max_tgt_len, args.n_train), args.batch_size)
	valid_x = BatchManager(load_data(VALID_X, src_vocab, max_src_len, args.n_valid), args.batch_size)
	valid_y = BatchManager(load_data(VALID_Y, tgt_vocab, max_tgt_len, args.n_valid), args.batch_size)

	model = Transformer(len(src_vocab), len(tgt_vocab), max_src_len, max_tgt_len,
			d_word_vec=300, N=6, n_head=3, d_q=100, d_k=100, d_v=100, d_model=300, d_inner=600,
			dropout=0.1, tgt_emb_prj_weight_share=True).cuda()

	model_file = args.model_file
	if os.path.exists(model_file):
		model.load_state_dict(torch.load(model_file))
		logging.info('Load model parameters from %s' % model_file)

	parameters = filter(lambda p : p.requires_grad, model.parameters())
	optimizer = torch.optim.Adam(parameters, lr=0.001)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.3)

	train(train_x, train_y, valid_x, valid_y, model, optimizer, scheduler, args.n_epochs)


if __name__ == '__main__':
	main()

