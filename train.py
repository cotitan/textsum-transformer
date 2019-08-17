import os
import shutil
import logging
import json
import torch
import argparse
from Transformer import Transformer, TransformerShareEmbedding
from tensorboardX import SummaryWriter
import utils
from utils import BatchManager, load_data, load_vocab, build_vocab
from pyrouge import Rouge155
from translate import greedy, print_summaries

parser = argparse.ArgumentParser(description='Selective Encoding for Abstractive Sentence Summarization in DyNet')

parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs [default: 3]')
parser.add_argument('--n_train', type=int, default=3803900,
                    help='Number of training data (up to 3803957 in gigaword) [default: 3803957]')
parser.add_argument('--n_valid', type=int, default=189651,
                    help='Number of validation data (up to 189651 in gigaword) [default: 189651])')
parser.add_argument('--batch_size', type=int, default=64, help='Mini batch size [default: 32]')
parser.add_argument('--ckpt_file', type=str, default='./ckpts/params_v2_9.pkl')
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

model_dir = './ckpts'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)


def run_batch(valid_x, valid_y, model):
    _, x = valid_x.next_batch()
    _, y = valid_y.next_batch()
    logits, _ = model(x, y)
    loss = model.loss_layer(logits.view(-1, logits.shape[-1]),
                            y[:, 1:].contiguous().view(-1))
    return loss


def eval_model(valid_x, valid_y, vocab, model):
    # if we put the following part outside the code,
    # error occurs
    r = Rouge155()
    r.system_dir = 'tmp/systems'
    r.model_dir = 'tmp/models'
    r.system_filename_pattern = "(\d+).txt"
    r.model_filename_pattern = "[A-Z].#ID#.txt"

    logging.info('Evaluating on a minibatch...')
    model.eval()
    _, x = valid_x.next_batch()
    with torch.no_grad():
        pred = greedy(model, x, vocab)
    _, y = valid_y.next_batch()
    y = y[:,1:].tolist()
    print_summaries(pred, vocab, 'tmp/systems', '%d.txt')
    print_summaries(y, vocab, 'tmp/models', 'A.%d.txt')

    try:
        output = r.convert_and_evaluate()
        output_dict = r.output_to_dict(output)
        logging.info('Rouge1-F: %f, Rouge2-F: %f, RougeL-F: %f'
                     % (output_dict['rouge_1_f_score'],
                        output_dict['rouge_2_f_score'],
                        output_dict['rouge_l_f_score']))
    except Exception as e:
        logging.info('Failed to evaluate')

    model.train()


def adjust_lr(optimizer, epoch):
	if (epoch + 1) % 2 == 0:
		# optimizer.param_groups[0]['lr'] *= math.sqrt((epoch+1)/10)
		optimizer.param_groups[0]['lr'] *= 0.5


def train(train_x, train_y, valid_x, valid_y, model,
          optimizer, tgt_vocab, scheduler, n_epochs=1, epoch=0):
    logging.info("Start to train with lr=%f..." % optimizer.param_groups[0]['lr'])
    n_batches = train_x.steps
    model.train()

    if os.path.isdir('runs/epoch%d' % epoch):
        shutil.rmtree('runs/epoch%d' % epoch)
    writer = SummaryWriter('runs/epoch%d' % epoch)
    i = epoch * train_x.steps
    for epoch in range(epoch, n_epochs):
        valid_x.bid = 0
        valid_y.bid = 0


        for idx in range(n_batches):
            optimizer.zero_grad()

            loss = run_batch(train_x, train_y, model)
            loss.backward()  # do not use retain_graph=True
            # torch.nn.utils.clip_grad_value_(model.parameters(), 5)

            optimizer.step()

            if (idx + 1) % 50 == 0:
                train_loss = loss.cpu().detach().numpy()
                model.eval()
                with torch.no_grad():
                    valid_loss = run_batch(valid_x, valid_y, model)
                logging.info('epoch %d: %d, training loss = %f, validation loss = %f'
                             % (epoch, idx + 1, train_loss, valid_loss))
                writer.add_scalar('scalar/train_loss', train_loss, i)
                writer.add_scalar('scalar/valid_loss', valid_loss, i)
                i += 1
                model.train()
            # if (idx + 1) % 2000 == 0:
            #     eval_model(valid_x, valid_y, tgt_vocab, model)
            # dump_tensors()

        adjust_lr(optimizer, epoch)
        save_state = {'state_dict': model.state_dict(),
                      'epoch': epoch + 1,
                      'lr': optimizer.param_groups[0]['lr']}
        torch.save(save_state, os.path.join(model_dir, 'params_v2_%d.pkl' % epoch))
        logging.info('Model saved in dir %s' % model_dir)
    writer.close()


def main():
    print(args)

    data_dir = '/home/disk3/tiankeke/sumdata/'
    TRAIN_X = os.path.join(data_dir, 'train/train.article.txt')
    TRAIN_Y = os.path.join(data_dir, 'train/train.title.txt')
    VALID_X = os.path.join(data_dir, 'train/valid.article.filter.txt')
    VALID_Y = os.path.join(data_dir, 'train/valid.title.filter.txt')

    src_vocab_file = 'sumdata/src_vocab.txt'
    if not os.path.exists(src_vocab_file):
        build_vocab([TRAIN_X], src_vocab_file)
    src_vocab = load_vocab(src_vocab_file, vocab_size=90000)
    
    tgt_vocab_file = 'sumdata/tgt_vocab.txt'
    if not os.path.exists(tgt_vocab_file):
        build_vocab([TRAIN_Y], tgt_vocab_file)
    tgt_vocab = load_vocab(tgt_vocab_file)

    # emb_file = '/home/tiankeke/workspace/embeddings/giga-vec1.bin'
    # vocab, embeddings = load_word2vec_embedding(emb_file)

    max_src_len = 100
    max_tgt_len = 40
    max_pos = 200

    bs = args.batch_size
    n_train = args.n_train
    n_valid = args.n_valid

    train_x = BatchManager(load_data(TRAIN_X, max_src_len, n_train), bs, src_vocab)
    train_y = BatchManager(load_data(TRAIN_Y, max_tgt_len, n_train), bs, tgt_vocab)
    train_x, train_y = utils.shuffle(train_x, train_y)

    valid_x = BatchManager(load_data(VALID_X, max_src_len, n_valid), bs, src_vocab)
    valid_y = BatchManager(load_data(VALID_Y, max_tgt_len, n_valid), bs, tgt_vocab)
    valid_x, valid_y = utils.shuffle(valid_x, valid_y)
    # model = Transformer(len(vocab), len(vocab), max_src_len, max_tgt_len, 1, 4, 256,
    #                     64, 64, 1024, src_tgt_emb_share=True, tgt_prj_wt_share=True).cuda()
    model = Transformer(len(src_vocab), len(tgt_vocab), max_pos, max_pos, 2, 4, 256,
                        1024, src_tgt_emb_share=False, tgt_prj_wt_share=True).cuda()
    # model = TransformerShareEmbedding(len(vocab), max_src_len, 2, 4,
    #                                   256, 1024, False, True).cuda()

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

    # eval_model(valid_x, valid_y, vocab, model)
    train(train_x, train_y, valid_x, valid_y, model,
          optimizer, tgt_vocab, scheduler, args.n_epochs, saved_state['epoch'])


if __name__ == '__main__':
    main()
    # TODO
    # 使用Pycharm,逐过程查看内部状态，看哪一步的结果值很小，可能是该步出问题

