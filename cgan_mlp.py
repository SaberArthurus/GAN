import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.init
from torch.nn.init import kaiming_normal

import os, time
import itertools
import pickle
import argparse

from torch.autograd import Variable
import torch.nn.init as init
from utils import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('-b', '--batch-size', default=10, type=int)
parser.add_argument('--decay-scale', default=0.1, type=float)
parser.add_argument('--decay-epoch', default=30, type=int)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float)
parser.add_argument('--print-freq', '-p', default=10, type=int)
parser.add_argument('--epochs', default=350, type=int)
parser.add_argument('--save-path', default='save/', type=str)
parser.add_argument('--evaluate', '-e')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')



def main():
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
            transforms.Scale(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform_train),
        batch_size=args.batch_size, shuffle=True)

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])


    # network
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader,
            g_model=G,
            g_optimizer=G_optimizer,
            d_model=D,
            d_optimizer=D_optimizer,
            criterion=criterion,
            epoch=epoch)

# Training
def train(train_loader, g_model, g_optimizer, d_model, d_optimizer, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()

if __name__ == '__main__':
    main()
