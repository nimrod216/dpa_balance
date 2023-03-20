import os
from Logger import Logger
from Datasets import CIFAR10, CIFAR100, ImageNet
from torch import nn
from collections import OrderedDict
from models.resnet import resnet18 as resnet18_imagenet
from models.resnet import resnet34 as resnet34_imagenet
from models.resnet import resnet50 as resnet50_imagenet
from models.resnet import resnet101 as resnet101_imagenet
from models.googlenet import googlenet as googlenet_imagenet
from models.inception import inception_v3 as inception_imagenet
from models.densenet import densenet121 as densenet_imagenet

basedir, _ = os.path.split(os.path.abspath(__file__))
basedir = os.path.join(basedir, 'data')

RESULTS_DIR = os.path.join(basedir, 'results')

DEBUG = False
USER_CMD = None
SEED = 123
BATCH_SIZE = 128
VERBOSITY = 0
INCEPTION = False

MODELS = {'9Layers': nn.Sequential(OrderedDict([
            ('L1', nn.Conv1d(in_channels=1, out_channels=2, kernel_size=19,
                             stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')),
            ('relu1', nn.ReLU()),
            ('L2', nn.Conv1d(in_channels=2, out_channels=3, kernel_size=9,
                             stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')),
            ('relu2', nn.ReLU()),
            ('L3', nn.Conv1d(in_channels=3, out_channels=4, kernel_size=7,
                             stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')),
            ('relu3', nn.ReLU()),
            ('L4', nn.Conv1d(in_channels=4, out_channels=5, kernel_size=5,
                             stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')),
            ('relu4', nn.ReLU()),
            ('L5', nn.Conv1d(in_channels=5, out_channels=6, kernel_size=5,
                             stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')),
            ('relu5', nn.ReLU()),
            ('L6', nn.Conv1d(in_channels=6, out_channels=7, kernel_size=5,
                             stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')),
            ('relu6', nn.ReLU()),
            ('L7', nn.Conv1d(in_channels=7, out_channels=8, kernel_size=5,
                             stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')),
            ('relu7', nn.ReLU()),
            ('L8', nn.Conv1d(in_channels=8, out_channels=9, kernel_size=5,
                             stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')),
            ('relu8', nn.ReLU()),
            ('L9', nn.Conv1d(in_channels=9, out_channels=10, kernel_size=5,
                             stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')),

        ])),
          '4Layers': nn.Sequential(OrderedDict([
            ('L1', nn.Conv1d(in_channels=1, out_channels=2, kernel_size=25,
                             stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')),
            ('relu1', nn.ReLU()),
            ('L2', nn.Conv1d(in_channels=2, out_channels=4, kernel_size=25,
                             stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')),
            ('relu2', nn.ReLU()),
            ('L3', nn.Conv1d(in_channels=4, out_channels=8, kernel_size=7,
                             stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')),
            ('relu3', nn.ReLU()),
            ('L4', nn.Conv1d(in_channels=8, out_channels=10, kernel_size=3,
                             stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')),

            ])),
          '7Layers': nn.Sequential(OrderedDict([
            ('L1', nn.Conv1d(in_channels=1, out_channels=2, kernel_size=21,
                             stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')),
            ('relu1', nn.ReLU()),
            ('L2', nn.Conv1d(in_channels=2, out_channels=4, kernel_size=15,
                             stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')),
            ('relu2', nn.ReLU()),
            ('L3', nn.Conv1d(in_channels=4, out_channels=6, kernel_size=15,
                             stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')),
            ('relu3', nn.ReLU()),
            ('L4', nn.Conv1d(in_channels=6, out_channels=7, kernel_size=3,
                             stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')),
            ('relu4', nn.ReLU()),
            ('L5', nn.Conv1d(in_channels=7, out_channels=8, kernel_size=3,
                             stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')),
            ('relu5', nn.ReLU()),
            ('L6', nn.Conv1d(in_channels=8, out_channels=9, kernel_size=3,
                             stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')),
            ('relu6', nn.ReLU()),
            ('L7', nn.Conv1d(in_channels=9, out_channels=10, kernel_size=3,
                             stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')),

            ]))}

DATASETS = {'cifar10':
                {'ptr': CIFAR10,  'dir': os.path.join(basedir, 'datasets')},
            'cifar100':
                {'ptr': CIFAR100, 'dir': os.path.join(basedir, 'datasets')},
            'imagenet':
                {'ptr': ImageNet, 'dir': '/mnt/ilsvrc2012'}}

LOG = Logger()
