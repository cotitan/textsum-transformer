import os
import logging
import json
import utils
import torch
import argparse
from Transformer import Transformer, TransformerShareEmbedding
from tensorboardX import SummaryWriter
from utils import BatchManager, load_data, get_vocab, build_vocab, load_word2vec_embedding

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
parser.add_argument('--ckpt_file', type=str, default='./models/params_v2_0.pkl')
args = parser.parse_args()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='log/train2.log',
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


def train(train_x, train_y, valid_x, valid_y, model, optimizer, scheduler, epochs=1, epoch=0):
    logging.info("Start to train...")
    n_batches = train_x.steps
    writer = SummaryWriter()
    model.train()
    for epoch in range(epoch, epochs):
        for idx in range(n_batches):
            optimizer.zero_grad()

            loss = run_batch(train_x, train_y, model)
            loss.backward()  # do not use retain_graph=True
            # torch.nn.utils.clip_grad_value_(model.parameters(), 5)

            optimizer.step()

            if (idx + 1) % 50 == 0:
                train_loss = loss.cpu().detach().numpy()
                model.eval()
                valid_loss = run_batch(valid_x, valid_y, model)
                logging.info('epoch %d, step %d, training loss = %f, validation loss = %f'
                             % (epoch, idx + 1, train_loss, valid_loss))
                writer.add_scalar('scalar/epoch_%d/train_loss' % epoch, train_loss, (idx+1)//50)
                writer.add_scalar('scalar/epoch_%d/valid_loss' % epoch, valid_loss, (idx+1)//50)
                model.train()
            del loss

        scheduler.step()
        save_state = {'state_dict': model.state_dict(),
                      'epoch': epoch + 1,
                      'lr': optimizer.param_groups[0]['lr']}
        torch.save(save_state, os.path.join(model_dir, 'params_v2_%d.pkl' % epoch))
        logging.info('Model saved in dir %s' % model_dir)
    writer.close()


def main():
    print(args)
    
    data_dir = '/home/tiankeke/workspace/datas/sumdata/'
    TRAIN_X = os.path.join(data_dir, 'train/train.article.txt')
    TRAIN_Y = os.path.join(data_dir, 'train/train.title.txt')
    VALID_X = os.path.join(data_dir, 'train/valid.article.filter.txt')
    VALID_Y = os.path.join(data_dir, 'train/valid.title.filter.txt')

    small_vocab_file = 'sumdata/small_vocab.json'
    if os.path.exists(small_vocab_file):
        small_vocab = json.load(open(small_vocab_file))
    else:
        small_vocab = build_vocab([TRAIN_X, TRAIN_Y], small_vocab_file, vocab_size=80000)

    emb_file = '/home/tiankeke/workspace/embeddings/giga-vec1.bin'
    vocab, embeddings = load_word2vec_embedding(emb_file)

    max_src_len = 101
    max_tgt_len = 47

    bs = args.batch_size

    train_x = BatchManager(load_data(TRAIN_X, vocab, max_src_len, args.n_train), bs)
    train_y = BatchManager(load_data(TRAIN_Y, vocab, max_tgt_len, args.n_train), bs)
    valid_x = BatchManager(load_data(VALID_X, vocab, max_src_len, args.n_valid), bs)
    valid_y = BatchManager(load_data(VALID_Y, vocab, max_tgt_len, args.n_valid), bs)

    model = Transformer(len(vocab), len(vocab), max_src_len, max_tgt_len, 1, 8, 512,
                        64, 64, 2048, src_tgt_emb_share=True, tgt_prj_emb_share=True,
                        embeddings=embeddings).cuda()

    # print(model)
    saved_state = {'epoch': 0, 'lr': 0.001}
    if os.path.exists(args.ckpt_file):
        saved_state = torch.load(args.ckpt_file)
        model.load_state_dict(saved_state['state_dict'])
        logging.info('Load model parameters from %s' % args.ckpt_file)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=saved_state['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.3)
    scheduler.step()  # last_epoch=-1, which will not update lr at the first time

    train(train_x, train_y, valid_x, valid_y, model, optimizer, scheduler, args.n_epochs, saved_state['epoch'])


if __name__ == '__main__':
    main()
    #TODO
    # 使用Pycharm,逐过程查看内部状态，看哪一步的结果值很小，可能是该步出问题

