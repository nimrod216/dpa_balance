import argparse
import datetime
import os
import numpy as np
import torch
import sys
import Config as cfg
from NeuralNet import NeuralNet
from Datasets import Datasets, DPA_datasets
import optuna

parser = argparse.ArgumentParser(description='Nimrod Admoni, nimrodadmoni@campus.technion.ac.il',
                                 formatter_class=argparse.RawTextHelpFormatter)

model_names = list(cfg.MODELS.keys())

parser.add_argument('-a', '--arch', metavar='ARCH', choices=cfg.MODELS.keys(), required=True,
                    help='model architectures and datasets:\n' + ' | '.join(cfg.MODELS.keys()))
# parser.add_argument('--action', choices=['QUANTIZE', 'INFERENCE'], required=True,
#                     help='QUANTIZE: symmetric min-max uniform quantization\n'
#                          'INFERENCE: either regular inference or hardware simulated inference')
# arch=resnet50, dataset=imagenet threads=[1],
# muxing=[0] epochs=100, cuda_conv=0
# LR=0.1 WD=0.0001 MOMENTUM=0.9 GAMMA=0.1
# NESTEROV=0 COSINE_LR=0 MILESTONES=[30, 60, 90] device=cuda data_type=bf16 stats=0 verbose=1 gpus=4 distributed=1 model_path=None
# self.train_scheduler_cuda = optim.lr_scheduler.MultiStepLR(self.cuda_optmizer, milestones=MILESTONES, gamma=GAMMA)

parser.add_argument('--desc',
                    help='additional string to the test')
parser.add_argument('--chkp', default=None, metavar='PATH',
                    help='model checkpoint')
parser.add_argument('--gpu', nargs='+', default=None,
                    help='GPU to run on (default: 0)')
parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                    help='batch size')
parser.add_argument('-v', '--verbosity', default=0, type=int,
                    help='verbosity level (0,1,2) (default: 0)')
parser.add_argument('--start_bits', choices=[2, 1, 0], default=0, type=int, metavar='N',
                    help='initial quantization bits')
parser.add_argument('--training_epoches', default=50, type=int, metavar='N',
                    help='initial quantization bits')
parser.add_argument('--lr', default=0.1, type=float,
                    help='learning rate')
parser.add_argument('--lr_plan', nargs='+', default=[30, 60, 90], type=int,
                    help='milestones')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='gamma factor for advancing lr')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='gamma factor for advancing lr')
parser.add_argument('--wd', default=0.0001, type=float,
                    help='gamma factor for advancing lr')
parser.add_argument('--skip_bn_recal', action='store_true',
                    help='skip BatchNorm recalibration (relevant only to the INFERENCE action)')
parser.add_argument('--quantize', default=0, type=int,
                    help='skip BatchNorm recalibration (relevant only to the INFERENCE action)')
parser.add_argument('--layers', default=4, type=int,
                    help='skip BatchNorm recalibration (relevant only to the INFERENCE action)')

args = parser.parse_args()


def train_network(arch, dataset, train_gen, test_gen, model_chkp=None, quantize=False, layers=4,
                  lr_plan=[30, 60, 90], desc=None, epoches=50, lr=0.1, gamma=0.1, momentum=0.9, wd=0):
    # Initialize log file
    name_str = '{}_train_network_size-{}'.format(arch, dataset)
    name_str = name_str + '_{}'.format(desc) if desc is not None else name_str
    name_str = name_str + '_seed-{}'.format(42)
    # Initialize model
    nn = NeuralNet(arch, dataset=dataset, model_chkp=model_chkp, quantize=quantize, layers=layers)
    # nn.model.set_min_max_update(True)
    nn.best_top1_acc = 0
    nn.next_train_epoch = 0
    loss = nn.train(train_gen=train_gen, test_gen=test_gen, epochs=epoches, desc=desc, lr=lr, lr_plan=lr_plan,
                    momentum=momentum, gamma=gamma, wd=wd)

    return loss


def main():
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    arch = args.arch
    dataset = 'dpa-dataset'
    saved_path, _ = os.path.split(os.path.abspath(__file__))
    saved_models_path = os.path.join(cfg.basedir, '{}_{}'.format(args.desc, datetime.date.today().strftime('%Y_%m_%d')))
    cfg.LOG.start_new_log(path=saved_models_path, name=args.desc)
    cfg.BATCH_SIZE = args.batch_size
    cfg.VERBOSITY = args.verbosity
    cfg.USER_CMD = ' '.join(sys.argv)
    cfg.INCEPTION = (arch == 'inception')

    dataset_ = DPA_datasets('/home/nimroda/dpa_balance/src/DATA_from_keyset_9.csv',factor=1000)

    # Deterministic random numbers
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    test_gen = dataset_.testset(batch_size=args.batch_size)
    train_gen = dataset_.trainset(batch_size=args.batch_size)

    model_chkp = None if args.chkp is None else cfg.RESULTS_DIR + '/' + args.chkp

    train_network(arch, dataset, train_gen, test_gen, model_chkp=model_chkp, quantize=args.quantize, layers=args.layers,
                  epoches=args.training_epoches, desc=args.desc, lr=args.lr, gamma=args.gamma, momentum=args.momentum,
                  wd=args.wd, lr_plan=args.lr_plan)
    return


def objective(trial):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.LOG.write("running on " + device)
    epochs = trial.suggest_categorical('epochs', [50, 80, 100, 150])
    batch_size = trial.suggest_categorical('batch_size', [8, 32, 128])
    lr = trial.suggest_loguniform('lr', 5e-5, 1e-2)
    arch = trial.suggest_categorical('arch', ['4Layers', '7Layers', '9Layers'])
    wd = trial.suggest_loguniform('wd', 1e-5, 1e-3)
    dataset = 'dpa-dataset'
    saved_path, _ = os.path.split(os.path.abspath(__file__))
    saved_models_path = os.path.join(cfg.basedir, '{}_{}'.format(args.desc, datetime.date.today().strftime('%Y_%m_%d')))
    cfg.LOG.start_new_log(path=saved_models_path, name=args.desc)
    cfg.BATCH_SIZE = args.batch_size
    cfg.VERBOSITY = args.verbosity
    cfg.USER_CMD = ' '.join(sys.argv)
    cfg.INCEPTION = (arch == 'inception')

    dataset_ = DPA_datasets('/home/nimroda/dpa_balance/src/DATA_from_keyset_9.csv')

    # Deterministic random numbers
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    test_gen = dataset_.testset(batch_size=batch_size)
    train_gen = dataset_.trainset(batch_size=batch_size)

    loss = train_network(arch, dataset, train_gen, test_gen, quantize=args.quantize,
                         epoches=epochs, desc=args.desc, lr=lr, gamma=args.gamma, momentum=args.momentum,
                         wd=wd, lr_plan=args.lr_plan)

    return loss


def parameter_sweep():
    cfg.LOG.start_new_log(name='parameter_search')
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)


if __name__ == '__main__':
    #parameter_sweep()
    main()