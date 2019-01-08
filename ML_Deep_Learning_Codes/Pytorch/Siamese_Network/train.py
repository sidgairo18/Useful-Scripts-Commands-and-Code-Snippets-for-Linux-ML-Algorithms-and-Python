# Code adapted from https://github.com/andreasveit/triplet-network-pytorch/blob/master/train.py
from __future__ import print_function, division
import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from triplet_dataloader import TripletImageLoader
from triplet_network import Tripletnet
from visdom import Visdom
import numpy as np

print ("Import Successful")

# Training Settings

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--test-batch_size', type=int, default=100, metavar='N', help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='helps enable cuda training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default:1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0.2, metavar='M', help='margin for triplet loss (default: 2)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint, default: None')
parser.add_argument('--name', default='TripletNet', type=str, help='name of experiment')

print ("Training Settings updated")

best_acc = 0

def main():

    global args, best_acc
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    global plotter
    #plotter = VisdomLinePlotter(env_name=args.name)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    #training loader
    train_loader = torch.utils.data.DataLoader(TripletImageLoader(base_path='.', filenames_filename='training_filename.txt', triplets_filename='training_triplet_filename.txt', transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size = args.batch_size, shuffle=True, **kwargs)
    
    #testing_loader - Remember to update filenames_filename, triplet_filename
    train_loader = torch.utils.data.DataLoader(TripletImageLoader(base_path='.', filenames_filename='training_filename.txt', triplets_filename='training_triplet_filename.txt', transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size = args.batch_size, shuffle=True, **kwargs)

    # Defining CNN architecture

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return x

    model = Net()
    tnet = Tripletnet(model)
    if args.cuda:
        tnet.cuda()

    print ("Model Initialized")


##################### Class for VisdomLinePlotter ##########################
class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name)
##################### Class for VisdomLinePlotter ##########################



if __name__ == "__main__":
    main()


