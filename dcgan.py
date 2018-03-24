import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os, time
import itertools
import pickle
import argparse

from models import *
from utils import *
from torch.autograd import Variable
import torch.nn.init as init



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('-b', '--batch-size', default=128, type=int)
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


class generator(nn.Module):
    def __init__(self, inplanes=128):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, inplanes*8, kernel_size=4, stride=1, padding=0)
        self.deconv1_bn = nn.BatchNorm2d(inplanes*8)

        self.deconv2 = nn.ConvTranspose2d(inplanes*8, inplanes*4, kernel_size=4, stride=2, padding=1)
        self.deconv2_bn = nn.BatchNorm2d(inplanes*4)

        self.deconv3 = nn.ConvTranspose2d(inplanes*4, inplanes*2, kernel_size=4, stride=2, padding=1)
        self.deconv3_bn = nn.BatchNorm2d(inplanes*2)

        self.deconv4 = nn.ConvTranspose2d(inplanes*2, inplanes, kernel_size=4, stride=2, padding=1)
        self.deconv4_bn = nn.BatchNorm2d(inplanes)

        self.deconv5 = nn.ConvTranspose2d(inplanes, 1, kernel_size=4, stride=2, padding=1)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal(m.weight, mode='fan_out')

    def forward(self, input):
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(input)))
        x = F.relu(self.deconv3_bn(self.deconv3(input)))
        x = F.relu(self.deconv4_bn(self.deconv4(input)))
        x = F.tanh(self.deconv5(x))

        return x

class discriminator(nn.Module):
    def __init__(self, inplanes=128):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, inplanes, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(inplanes, inplanes*2, kernel_size=4, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(inplanes*2)
        self.conv3 = nn.Conv2d(inplanes*2, inplanes*4, kernel_size=4, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(inplanes*4)
        self.conv4 = nn.Conv2d(inplanes*8, kernel_size=4, stride=2, padding=1)
        self.conv4_bn = nn.BatchNorm2d(inplanes*8)
        self.conv5 = nn.Conv2d(inplanes*8, 1, kernel_size=4, stride=1, padding=0)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_normal(m.weight, mode='fan_out')

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x



def main():
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    image_size = 64
    transform_train = transforms.Compose([
            transforms.Scale(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    

    trainset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size, shuffle=True)

    # network
    G = generator(128)
    D = discriminator(128)
    G.weight_init()
    D.weight_init()
    G.cuda()
    D.cuda()

    criterion = nn.BCELoss()
    G_optimizer = optim.Adam(G.parametsers(), lr=args.lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parametsers(), lr=args.lr, betas=(0.5, 0.999))

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loadertrain_loader, 
            g_model=G, 
            g_optimizer=G_optimizer,
            d_model=D, 
            d_optimizer=d_optimizer, 
            criterion=criterion, 
            epoch=epoch)

# Training
def train(train_loader, g_model, g_optimizer, d_model, d_optimizer, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    D_losses = AverageMeter()
    G_losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, _) in enumerate(train_loader):
        # train discriminator D
        d_model.zero_grad()
        data_time.update(time.time() - end)

        mini_batch = input.size()[0]
        
        yreal = torch.ones(mini_batch)
        yfake = torch.ones(mini_batch)
        input_var, yreal_var, yfake_var = Variable(input.cuda()), Variable(yreal_var.cuda()), Variable(yfake_var.cuda())
        d_result = d_model(input_var).squeeze()
        d_real_loss = criterion(d_result, yreal_var)

        z = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z = Variable(z.cuda())
        g_result = g_model(z)

        d_result = d(g_result).squeeze()
        d_fake_loss = criterion(d_result, yfake_var)
        d_fake_score = d_result.data.mean()

        d_train_loss = d_real_loss + d_fake_loss

        d_train_loss.backward()
        d_optimizer.step()

        D_losses.update(d_train_loss[0], input.size(0))


        # train generator G
        G.zero_grad()

        z = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z = Variable(z.cuda())
        g_result = g_model(z)
        d_result = d_model(g_result).squeeze()
        g_train_loss = criterion(d_result, yreal_var)
        g_train_loss.backward()
        g_optimizer.step()
        G_losses.update(g_train_loss[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
if name == '__main__':
    main()