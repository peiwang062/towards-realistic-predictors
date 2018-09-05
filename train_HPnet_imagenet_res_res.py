import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
import datasets
import models as models
import matplotlib.pyplot as plt
import torchvision.models as torch_models
from extra_setting import *
import scipy.io as sio

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('-d', '--dataset', default='imagenet', help='dataset name')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-c', '--channel', type=int, default=16,
                    help='first conv channel (default: 16)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--gpu', default='4,5,6,7', help='index of gpus to use')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lr_step', default='1', help='decreasing strategy')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # select gpus
    args.gpu = args.gpu.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    # data loader
    assert callable(datasets.__dict__[args.dataset])
    get_dataset = getattr(datasets, args.dataset)
    num_classes = datasets._NUM_CLASSES[args.dataset]
    train_loader, val_loader = get_dataset(
        batch_size=args.batch_size, num_workers=args.workers)

    # create model
    model_main = torch_models.resnet50(pretrained=True)
    model_main = torch.nn.DataParallel(model_main, device_ids=range(len(args.gpu))).cuda()
    model_hp = models.__dict__['hp_net_res50']()
    model_hp = torch.nn.DataParallel(model_hp, device_ids=range(len(args.gpu))).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_f = nn.CrossEntropyLoss(reduce=False).cuda()

    optimizer_m = torch.optim.SGD(model_main.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
    optimizer_h = torch.optim.Adam(model_hp.parameters(), lr=0.00001, weight_decay=1e-3)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model_main.load_state_dict(checkpoint['state_dict_m'])
            model_hp.load_state_dict(checkpoint['state_dict_h'])
            optimizer_m.load_state_dict(checkpoint['optimizer_m'])
            optimizer_h.load_state_dict(checkpoint['optimizer_h'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        validate(val_loader, model_main, criterion)
        return

    lr_step = list(map(int, args.lr_step.split(',')))

    for epoch in range(args.start_epoch, args.epochs):
        if epoch in lr_step:
            for param_group in optimizer_m.param_groups:
                param_group['lr'] *= 0.1
            for param_group in optimizer_h.param_groups:
                param_group['lr'] *= 0.1

        # train for one epoch
        train(train_loader, model_main, model_hp, optimizer_m, optimizer_h, epoch, criterion_f)

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model_main, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict_m': model_main.state_dict(),
            'state_dict_h': model_hp.state_dict(),
            'best_prec1': best_prec1,
            'optimizer_m': optimizer_m.state_dict(),
            'optimizer_h': optimizer_h.state_dict(),
        }, is_best)

        save_predicted_hardness(train_loader, val_loader, model_hp)


def train(train_loader, model_main, model_hp, optimizer_m, optimizer_h, epoch, criterion_f):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_m = AverageMeter()
    losses_h = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model_main.train()
    model_hp.train()

    end = time.time()
    for i, (input, target, index) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda(async=True)

        # compute output
        predicted_labels = model_main(input)
        loss_m = criterion_f(predicted_labels, target).squeeze()
        predicted_hardness_scores = model_hp(input).squeeze()
        loss_m = torch.mean(loss_m * predicted_hardness_scores)
        loss_h = opposite_loss(predicted_labels, predicted_hardness_scores, target, criterion_f)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(predicted_labels, target, topk=(1, 5))
        losses_m.update(loss_m.item(), input.size(0))
        losses_h.update(loss_h.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer_m.zero_grad()
        loss_m.backward(retain_graph=True)
        optimizer_m.step()

        optimizer_h.zero_grad()
        loss_h.backward()
        optimizer_h.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            curr_lr_m = optimizer_m.param_groups[0]['lr']
            curr_lr_h = optimizer_h.param_groups[0]['lr']
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: [{4}][{5}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_m {loss_m.val:.4f} ({loss_m.avg:.4f})\t'
                  'Loss_h {loss_h.val:.4f} ({loss_h.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, args.epochs, i, len(train_loader), curr_lr_m, curr_lr_h,
                batch_time=batch_time, data_time=data_time, loss_m=losses_m, loss_h=losses_h, top1=top1, top5=top5))


def validate(val_loader, model_main, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model_main.eval()

    end = time.time()
    for i, (input, target, index) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda(async=True)

        # compute output
        output = model_main(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='./imagenet/checkpoint_res_res4.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './imagenet/model_best_res_res4.pth.tar')


def save_predicted_hardness(train_loader, val_loader, model_hp):
    model_hp.eval()

    hardness_scores_tr = []
    hardness_scores_idx_tr = []
    for i, (input, target, index) in enumerate(train_loader):
        input_var = torch.autograd.Variable(input, volatile=True)
        predicted_hardness_scores = model_hp(input_var).squeeze()
        scores = predicted_hardness_scores.data.cpu().numpy()
        hardness_scores_tr = np.concatenate((hardness_scores_tr, scores), axis=0)
        index = index.numpy()
        hardness_scores_idx_tr = np.concatenate((hardness_scores_idx_tr, index), axis=0)
    sio.savemat('./imagenet/hardness_scores_res_res_tr4.mat', {'hardness_scores_tr': hardness_scores_tr})
    sio.savemat('./imagenet/hardness_scores_idx_res_res_tr4.mat', {'hardness_scores_idx_tr': hardness_scores_idx_tr})

    hardness_scores_val = []
    hardness_scores_idx_val = []
    for i, (input, target, index) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, volatile=True)
        predicted_hardness_scores = model_hp(input_var).squeeze()
        scores = predicted_hardness_scores.data.cpu().numpy()
        hardness_scores_val = np.concatenate((hardness_scores_val, scores), axis=0)
        index = index.numpy()
        hardness_scores_idx_val = np.concatenate((hardness_scores_idx_val, index), axis=0)
    sio.savemat('./imagenet/hardness_scores_res_res_val4.mat', {'hardness_scores_val': hardness_scores_val})
    sio.savemat('./imagenet/hardness_scores_idx_res_res_val4.mat', {'hardness_scores_idx_val': hardness_scores_idx_val})

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
