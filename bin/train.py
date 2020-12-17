import sys
import os
import argparse
import logging
import json
import time
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD #优化算法库
from torch.nn import DataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from tensorboardX import SummaryWriter
#from model.centers_loss import CenterLoss

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from dataset.afew import AFEWDataset
from model.vgg import ERNet
import model.loss_functional as loss_functional

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help='Path to the config file in json format')
parser.add_argument('save_path', default=None, metavar='SAVE_PATH', type=str,
                    help='Path to the saved models')
parser.add_argument('--num_workers', default=1, type=int, help='''Number of
                    workers for each data loader''')
parser.add_argument('--num_gpu', default=1, type=int, help='''Number of GPU''')
parser.add_argument('--resume', default=0, type=int, help='''If resume from
                    previous run''')

def loss_func(out, target, loss_type):
    if loss_type == 'L1':
        return loss_functional.loss_l1(out, target)
    if loss_type == 'SL1':
        return loss_functional.loss_sl1(out, target)
    elif loss_type == 'L2':
        return loss_functional.loss_l2(out, target)
    elif loss_type == 'crossEntropy':
        return loss_functional.loss_crossEn(out, target)
    else:
        raise Exception('Unknown loss type : {}'.format(loss_type))

def lr_schedule(lr, lr_factor, epoch_now, lr_epochs):
    count = 0
    for epoch in lr_epochs:
        if epoch_now >= epoch:
            count += 1
            continue
        break

    return lr * np.power(lr_factor, count)

def train_epoch(summary, summart_writer, cfg, model, optimizer, dataloader):
    torch.set_grad_enabled(True)
    model.train()
    device = torch.device('cuda')

    steps = len(dataloader)
    dataiter = iter(dataloader)
    # batch_size = dataloader.batch_size
    # imdb = dataloader.dataset.imdb

    time_now = time.time()
    loss_sum = 0
    loss_train_sum = 0

    predict_dict = {}
    label_dict = {}
    for step in range(steps):
        img, vid_name, label, idcs = next(dataiter)
        img = img.to(device)

        out, attention_weights = model(img)

        #model.centers_loss = CenterLoss(num_classes=7, feat_dim=2, use_gpu=True)
        #model.optimizer_centloss = torch.optim.SGD(model.centers_loss.parameters(), lr=0.5)

        loss = loss_func(out, label.to(device), cfg['TRAIN.LOSS'])
        loss_sum += loss.item()
        loss_train_sum += loss_sum

        #loss = model.centers_loss(model.features, labels) * model.alpha + loss
        #optimizer.zero_grad()
        #loss.backward()
        #for param in model.centers_loss.parameters():
            # lr_cent is learning rate for center loss, e.g. lr_cent = 0.5
            #param.grad.data *= (model.lr_cent / (model.alpha * model.lr))
        #optimizer.step()

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg['grad_norm'])
        optimizer.step()

        for i in range(label.shape[0]):
            predict_dict[vid_name[i]] = predict_dict.get(vid_name[i], 0) + out.detach()[i]
            label_dict[vid_name[i]] = label_dict.get(vid_name[i], label[i])

        summary['step'] += 1

        if summary['step'] % cfg['log_every'] == 0:
            time_spend = time.time() - time_now
            time_now = time.time()

            loss_sum /= cfg['log_every']

            predicts = []
            labels = []
            for vid, score in predict_dict.items():
                predicts.append(score.argmax().item())
                labels.append(label_dict[vid].argmax().item())

            # cm = confusion_matrix(labels, predicts)
            f1 = f1_score(labels, predicts, average='macro')
            acc = accuracy_score(labels, predicts)

            logging.info('{}, Train, Epoch: {}, step: {}, LR:{}, Loss: {:.7f}, f1:{}, acc:{}, '
                         ' Run Time: {:.2f} sec'
                         .format(time.strftime('%Y-%m-%d %H:%M:%S'),
                                 summary['epoch']+1,
                                 summary['step'],
                                 optimizer.defaults['lr'],
                                 loss_sum,
                                 f1,
                                 acc,
                                 # cm,
                                 time_spend))

            predict_dict = {}
            label_dict = {}

            summart_writer.add_scalar('train/loss', loss_sum, summary['step'])

            loss_sum = 0
    summary['epoch'] += 1

    return summary

def valid_epoch(model, dataloader):
    torch.set_grad_enabled(False)
    model.eval()
    device = torch.device('cuda')

    steps = len(dataloader)
    dataiter = iter(dataloader)
    # batch_size = dataloader.batch_size
    # imdb = dataloader.dataset.imdb

    predict_dict = {}
    label_dict = {}
    # start = time.time()
    for step in range(steps):
        img, vid_name, label, idcs = next(dataiter)
        img = img.to(device)
        out, attention_weights = model(img)

        for i in range(label.shape[0]):
            predict_dict[vid_name[i]] = predict_dict.get(vid_name[i], 0) + out[i]
            label_dict[vid_name[i]] = label_dict.get(vid_name[i], label[i])
    # end = time.time()
    # print("用时:",end-start)

    predicts = []
    labels = []
    for vid, score in predict_dict.items():
        predicts.append(score.argmax().item())
        labels.append(label_dict[vid].argmax().item())

    cm = confusion_matrix(labels, predicts)
    f1 = f1_score(labels, predicts,average='macro')
    acc = accuracy_score(labels, predicts)

    return cm, f1, acc


def run(args):
    with open(args.cfg_path) as f:
        cfg = json.load(f) #从文件中读取json字符串

    logging.info(cfg) #通常只记录关键节点信息，用于确认一切都是按照我们预期的那样进行工作,创建一条严重级别为INFO的日志记录

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path) #创建一级目录

    # if not args.resume:
    with open(os.path.join(args.save_path, 'cfg.json'), 'w') as f: #将多个路径组合后返回,w可创建文件
        json.dump(cfg, f, indent=1) #indent代表缩进，对于文件编码json

    num_devices = torch.cuda.device_count()
    if num_devices < args.num_gpu:
        raise Exception('#available gpu : {} < --num_gpu: {}'
                        .format(num_devices, args.num_gpu))
    device = torch.device('cuda')
    model = DataParallel(ERNet(), device_ids=range(args.num_gpu)).to(device).train()
    optimizer = SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
                    weight_decay=cfg['weight_decay'])

    dataloader_train = DataLoader(AFEWDataset(cfg, '1'),
                                  batch_size=cfg['TRAIN.BATCH_SIZE'],
                                  num_workers=args.num_workers,
                                  drop_last=False,
                                  shuffle=True)
    sta_train = dataloader_train.dataset.statistic
    logging.info(sta_train)

    dataloader_val = DataLoader(AFEWDataset(cfg, '3'),
                                  batch_size=cfg['VALID.BATCH_SIZE'],
                                  num_workers=args.num_workers,
                                  drop_last=False,
                                  shuffle=False)

    sta_val = dataloader_val.dataset.statistic
    logging.info(sta_val)

    summary_train = {'epoch':0, 'step':0}
    summart_writer = SummaryWriter(args.save_path)

    epoch_start = 0
    acc_best = 0

    if args.resume:
        ckpt_path = os.path.join(args.save_path, 'best.ckpt')
        ckpt = torch.load(ckpt_path)
        model.module.load_state_dict(ckpt['state_dict'])
        summary_train = {'epoch': ckpt['epoch'], 'step': ckpt['step']}
        acc_best = ckpt['acc_best']
        epoch_start = ckpt['epoch']

    schedule = StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(epoch_start, cfg['epoch']):
        summary_train = train_epoch(summary_train, summart_writer, cfg, model, optimizer, dataloader_train)

        cm, f1, acc = valid_epoch(model, dataloader_val)

        schedule.step()

        logging.info('{}, Valid, Epoch:{}, Step: {}, F1:{}, Acc:{}, CM:{}'
                     .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                             summary_train['epoch'],
                             summary_train['step'],
                             f1, acc, cm))

        if acc > acc_best:
            acc_best = acc
            torch.save({'epoch': summary_train['epoch'],
                        'step': summary_train['step'],
                        'state_dict': model.module.state_dict(),
                        'acc_best':acc_best},
                       os.path.join(args.save_path, 'best.ckpt'))

        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'state_dict': model.module.state_dict(),
                    'acc_best': acc_best},
                   os.path.join(args.save_path, 'train.ckpt'))

    summart_writer.close()

def main():
    logging.basicConfig(level=logging.INFO) #可以通过设置不同的日志等级，在release版本中只输出重要信息，而不必显示大量的调试信息，INFO：确认一切按预期运行

    args = parser.parse_args() #args子类实例
    run(args)

if __name__ == '__main__':
    main()
    # ckpt = torch.load('/home/ubuntu/Downloads/facial_expression/CODE/self_atttention_for_ER/result/best0.ckpt')
    # sd = ckpt['state_dict']
    # # summary_train = {'epoch': ckpt['epoch'], 'step': ckpt['step']}
    # # acc_best = ckpt['acc_best']
    # # epoch_start = ckpt['epoch']
    # torch.save({'epoch': ckpt['epoch'],
    #             'step': ckpt['step'],
    #             'state_dict': sd,
    #             'acc_best': 0.3611},
    #            '/home/ubuntu/Downloads/facial_expression/CODE/self_atttention_for_ER/result/best.ckpt')
    # pass

