import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
from utils import set_manual_seed, init_weights
from model import TextCNN, TextLSTM, TextTransformer, TextDPCNN
from dataset import CharLevelDataset, WordLevelDataset
import argparse

def str2bool(v):
    if int(v):
        return True
    return False

def parse_config():
    parser = argparse.ArgumentParser(description="Run sentiment clasification.")
    parser.add_argument('--mode', type=str, choices=["train", "eval"])
    parser.add_argument('--model_type', type=int, choices=range(1,5), default=4) # 1-4: "cnn", "lstm", "trans", "dpcnn"
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--valid_interval', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--vocab_size', type=int, default=256)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128, help='channels for CNN, hidden_size for LSTM, Transformer or DPCNN')
    parser.add_argument('--num_layers', type=int, default=1, help='layers for LSTM, Transformer or DPCNN')
    parser.add_argument('--bi_flag', type=str2bool, default=False, help='bidirection flag for LSTM')
    parser.add_argument('--norm_flag', type=str2bool, default=False, help='batch_norm flag for DPCNN')
    parser.add_argument('--output_size', type=int, default=2)

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument('--word_level', default=False, action='store_true')
    parser.add_argument('--vocab_file', type=str, default='./conf/word_vocab.pkl')
    parser.add_argument('--stop_file', type=str, default='./conf/stopwords.txt')

    parser.add_argument('--train_path', type=str)
    parser.add_argument('--valid_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--model_file', type=str)

    return parser.parse_args()

def build_model(args):
    if args.model_type == 1:
        model = TextCNN(args.vocab_size, args.embed_size, args.hidden_size, [2,3,4,5], args.output_size)
    elif args.model_type == 2:
        model = TextLSTM(args.vocab_size, args.embed_size, args.hidden_size, args.num_layers, args.output_size, args.bi_flag)
    elif args.model_type == 3:
        model = TextTransformer(args.vocab_size, args.embed_size, args.hidden_size, args.num_layers, args.output_size)
    elif args.model_type == 4:
        model = TextDPCNN(args.vocab_size, args.embed_size, args.hidden_size, args.num_layers, args.output_size, args.norm_flag)
    return model

def prepare_dataset(args, file_path, shuffle):
    if args.word_level:
        dataset = WordLevelDataset(file_path, args.vocab_file, args.stop_file, max_len=args.seq_len)
    else:
        dataset = CharLevelDataset(file_path, max_len=args.seq_len)
    assert(dataset.get_vocab_size() == args.vocab_size)
    dataset = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, pin_memory=True, num_workers=0)
    return dataset

def train(args):
    train_set = prepare_dataset(args, args.train_path, True)
    valid_set = prepare_dataset(args, args.valid_path, False)

    use_cuda = torch.cuda.is_available()
    gpu = 'cuda:%d' % args.device
    device = torch.device(gpu if use_cuda else "cpu")

    model = build_model(args)
    #model.apply(init_weights)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.NLLLoss()
    max_valid_acc = 0
    for epoch in range(args.max_epoch):
        t1 = time.time()
        model.train()
        epoch += 1
        losses = []
        acc_count = 0
        total_count = 0
        for x,y in train_set:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            pred = torch.argmax(y_pred, 1)
            acc_count += (pred.cpu() == y.cpu()).sum().item()
            total_count += len(y)

        avg_loss = np.mean(losses)
        avg_acc = acc_count / total_count
        print("Epoch:{} train time:{:.2f} loss:{:.5f} acc:{:.5f}".format(epoch, time.time()-t1, avg_loss, avg_acc))
        if epoch % args.valid_interval == 0:
            t1 = time.time()
            model.eval()
            with torch.no_grad():
                y_true = torch.LongTensor()
                y_pred = torch.LongTensor()
                losses = []
                for x,y in valid_set:
                    y_true = torch.cat((y_true, y), 0)
                    x,y = x.to(device), y.to(device)
                    out = model(x)
                    loss = loss_fn(out, y)
                    losses.append(loss.item())
                    pred = torch.argmax(out, 1)
                    y_pred = torch.cat((y_pred, pred.cpu()), 0)

                acc = (y_pred == y_true).sum().item()
                acc = acc / y_true.shape[0]
                valid_loss = np.mean(losses)
                print("Epoch:{} valid time:{:.2f} loss:{:.5f} acc:{:.5f}".format(epoch, time.time()-t1, valid_loss, acc))
                if acc > max_valid_acc:
                    max_valid_acc = acc
                    #save_file = "%s.%d.pth" % (args.model_file, epoch)
                    save_file = args.model_file
                    #state_dict = model.module.state_dict() #if use torch.nn.DataParallel
                    state_dict = model.state_dict()
                    torch.save(state_dict, save_file)

def eval(args):
    test_set = prepare_dataset(args, args.test_path, False)

    use_cuda = torch.cuda.is_available()
    gpu = 'cuda:%d' % args.device
    device = torch.device(gpu if use_cuda else "cpu")

    model = build_model(args)
    model.load_state_dict(torch.load(args.model_file, map_location=device))
    model.to(device)
    model.eval()

    t1 = time.time()
    with torch.no_grad():
        y_true = torch.LongTensor()
        y_pred = torch.LongTensor()
        for x,y in test_set:
            x = x.to(device)
            out = model(x)
            pred = torch.argmax(out, 1)
            y_pred = torch.cat((y_pred, pred.cpu()), 0)
            y_true = torch.cat((y_true, y), 0)

        acc = (y_pred == y_true).sum().item()
        acc = acc / y_true.shape[0]
        key = 'char_level'
        if args.word_level:
            key = 'word_level'
        print("test time:{:.2f} {}_model:{} acc:{:.5f}".format(time.time()-t1, key, args.model_type, acc))

if __name__ == "__main__":
    set_manual_seed(2021)
    args = parse_config()
    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        eval(args)
    else:
        print("wrong mode: {}".format(args.mode))
