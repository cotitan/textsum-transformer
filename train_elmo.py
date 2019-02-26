import os
import shutil
import logging
import json
import torch
import argparse
from Transformer import ElmoTransformer
from tensorboardX import SummaryWriter
from utils import BatchManager, load_data, get_vocab, build_vocab, load_word2vec_embedding, dump_tensors
from pyrouge import Rouge155
from translate import greedy, print_summaries

parser = argparse.ArgumentParser(description='Selective Encoding for Abstractive Sentence Summarization in DyNet')

parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs [default: 3]')
parser.add_argument('--n_train', type=int, default=3803900,
                    help='Number of training data (up to 3803957 in gigaword) [default: 3803957]')
parser.add_argument('--n_valid', type=int, default=189651,
                    help='Number of validation data (up to 189651 in gigaword) [default: 189651])')
parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size [default: 32]')
parser.add_argument('--ckpt_file', type=str, default='./models/params_elmo_0.pkl')
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='log/train_elmo.log',
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


def run_batch(batch_x, batch_y, batch_y_ids, model):
    x = batch_x.next_batch()
    y = batch_y.next_batch()
    y_ids = batch_y_ids.next_batch().cuda()
    # max_len = max(len(seq) for seq in y)
    # logits = torch.zeros(len(x), 0, model.n_vocab).cuda()
    # for i in range(1, max_len):
    #     logit = model(x, [seq[:i] for seq in y])  # [batch, seqlen, n_vocab]
    #     logits = torch.cat([logits, logit[:,-1,:].unsqueeze(1)], dim=1)
    logits = model(x, y)
    loss = model.loss_layer(logits.view(-1, logits.shape[-1]),
                            y_ids[:, 1:].contiguous().view(-1))
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
    x = valid_x.next_batch()
    with torch.no_grad():
        pred = greedy(model, x, vocab)
    y = valid_y.next_batch()[:,1:].tolist()
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


def train(train_x, train_y, y_ids, valid_x, valid_y, valid_y_ids, model,
          optimizer, vocab, scheduler, n_epochs=1, epoch=0):
    logging.info("Start to train with lr=%f..." % optimizer.param_groups[0]['lr'])
    n_batches = train_x.steps
    model.train()
    for epoch in range(epoch, n_epochs):
        valid_x.bid = 0
        valid_y.bid = 0
        valid_y_ids.bid = 0

        writer_dir = 'runs/epoch%d' % epoch
        if os.path.isdir(writer_dir):
            shutil.rmtree(writer_dir)
        writer = SummaryWriter(writer_dir)

        for idx in range(n_batches):
            optimizer.zero_grad()

            loss = run_batch(train_x, train_y, y_ids, model)
            loss.backward()  # do not use retain_graph=True
            # torch.nn.utils.clip_grad_value_(model.parameters(), 5)

            optimizer.step()

            if (idx + 1) % 50 == 0:
                train_loss = loss.cpu().detach().numpy()
                model.eval()
                with torch.no_grad():
                    valid_loss = run_batch(valid_x, valid_y, valid_y_ids, model)
                logging.info('epoch %d, step %d, training loss = %f, validation loss = %f'
                             % (epoch, idx + 1, train_loss, valid_loss))
                writer.add_scalar('scalar/train_loss', train_loss, (idx + 1) // 50)
                writer.add_scalar('scalar/valid_loss', valid_loss, (idx + 1) // 50)
                model.train()
                # torch.cuda.empty_cache()
            # if (idx + 1) % 2000 == 0:
            #     eval_model(valid_x, valid_y, vocab, model)
            # dump_tensors()

        if epoch < 4:
            scheduler.step()  # make sure lr will not be too small
        save_state = {'state_dict': model.state_dict(),
                      'epoch': epoch + 1,
                      'lr': optimizer.param_groups[0]['lr']}
        torch.save(save_state, os.path.join(model_dir, 'params_elmo_%d.pkl' % epoch))
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

    # emb_file = '/home/tiankeke/workspace/embeddings/giga-vec1.bin'
    # vocab, embeddings = load_word2vec_embedding(emb_file)

    max_src_len = 101
    max_tgt_len = 47

    bs = args.batch_size

    vocab = small_vocab

    train_x = BatchManager(load_data(TRAIN_X, max_src_len, args.n_train), bs, pad=False)
    train_y = BatchManager(load_data(TRAIN_Y, max_tgt_len, args.n_train), bs, pad=False)
    y_ids = BatchManager(load_data(TRAIN_Y, max_tgt_len, args.n_train, vocab), bs)
    valid_x = BatchManager(load_data(VALID_X, max_src_len, args.n_valid), bs, pad=False)
    valid_y = BatchManager(load_data(VALID_Y, max_tgt_len, args.n_valid), bs, pad=False)
    valid_y_ids = BatchManager(load_data(VALID_Y, max_tgt_len, args.n_valid, vocab), bs)

    # model = Transformer(len(vocab), len(vocab), max_src_len, max_tgt_len, 1, 4, 256,
    #                     64, 64, 1024, src_tgt_emb_share=True, tgt_prj_emb_share=True).cuda()
    # model = Transformer(len(vocab), len(vocab), max_src_len, max_tgt_len, 1, 6, 300,
    #                     50, 50, 1200, src_tgt_emb_share=True, tgt_prj_emb_share=True).cuda()
    model = ElmoTransformer(max_src_len, len(vocab), 1, 4, 64, 64, 256, 1024).cuda()

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
    train(train_x, train_y, y_ids, valid_x, valid_y, valid_y_ids, model,
          optimizer, vocab, scheduler, args.n_epochs, saved_state['epoch'])


if __name__ == '__main__':
    main()
    # TODO
    # 使用Pycharm,逐过程查看内部状态，看哪一步的结果值很小，可能是该步出问题

