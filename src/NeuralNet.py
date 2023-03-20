import torch
import torch.nn as nn
import time
import os
import Config as cfg
from StatsLogger import StatsLogger
from collections import OrderedDict
from CustomLoss import InfLoss

class NeuralNet:
    def __init__(self, arch, dataset, model_chkp=None, pretrained=True, quantize=False, layers=4):
        """
        NeuralNet class wraps a model architecture and adds the functionality of training and testing both
        the entire model and the prediction layers.

        :param arch: a string that represents the model architecture, e.g., 'alexnet'
        :param dataset: a string that represents the dataset, e.g., 'cifar100'
        :param model_chkp: a model checkpoint path to be loaded (default: None)
        :param pretrained: whether to load PyTorch pretrained parameters, used for ImageNet (default: True)
        """
        cfg.LOG.write('__init__: arch={}, dataset={}, model_chkp={}, pretrained={}'
                      .format(arch, dataset, model_chkp, pretrained))

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            cfg.LOG.write('Using GPU')
        else:
            self.device = torch.device('cpu')
            cfg.LOG.write('WARNING: Found no valid GPU device - Running on CPU')
        self.arch = '{}_{}'.format(arch, dataset)
        self.model = cfg.MODELS[arch]
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        # self.parallel_model = torch.nn.DataParallel(self.model).cuda(self.device)
        # Disabled parallel model. Statistics collection does not support GPU parallelism.
        self.parallel_model = self.model.cuda(self.device)
        #self.criterion = torch.nn.MSELoss().cuda(self.device)
        self.criterion = InfLoss().cuda(self.device)
        self.quantize = quantize
        if self.quantize:
            raise NotImplementedError
        self.optimizer = None
        self.lr_plan = None
        self.best_top1_acc = 0
        self.next_train_epoch = 0
        self.plato_epochs = 0
        if model_chkp is not None and model_chkp is not 'pretrained':
            self._load_state(model_chkp)

        self.stats = StatsLogger()
        # self.print_statistics = print_statistics

    def test(self, test_gen, iterations=None):
        batch_time = self.AverageMeter('Time', ':6.3f')
        losses = self.AverageMeter('Loss', ':.4e')
        progress = self.ProgressMeter(len(test_gen), batch_time, losses, prefix='Test: ')

        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(test_gen):

                input = input.cuda(self.device, non_blocking=True)
                target = target.cuda(self.device, non_blocking=True)
                # new_shape = [1] + list(input.size())
                # input = torch.reshape(input, new_shape)
                # Compute output
                output = self.parallel_model(input)
                output = torch.flatten(output, 1)
                loss = self.criterion(output, target)
                #loss = torch.max(torch.abs(output-target))

                # Measure accuracy and record loss
                losses.update(loss.item(), input.size(0))


                # Measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # Print to screen
                if i % 10 == 0:
                    progress.print(i)

                if iterations is not None:
                    if i == iterations:
                        break

            # TODO: this should also be done with the ProgressMeter
            cfg.LOG.write(' * Loss {losses.avg:.3f}'.format(losses=losses))

        return losses.avg

    def train(self, train_gen, test_gen, epochs, lr=0.0001, lr_plan=None, momentum=0.9, wd=5e-4, gamma=0.1,
              desc=None, iterations=None):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        self.lr_plan = lr_plan

        cfg.LOG.write(
            'train_pred: epochs={}, lr={}, lr_plan={}, momentum={}, wd={}, batch_size={}, optimizer={}, quantization={}'
            .format(epochs, lr, lr_plan, momentum, wd, cfg.BATCH_SIZE, 'SGD', self.quantize))

        for epoch in range(self.next_train_epoch, epochs):
            self._adjust_lr_rate(self.optimizer, epoch, lr_plan, gamma)
            # self.parallel_model.quant_size(self.quant)
            # A precautions that when the learning rate is 0 then no parameters are updated
            lr = self.optimizer.param_groups[0]['lr']
            if lr == 0:
                cfg.LOG.write('lr=0, running train steps with no_grad()')
                with torch.no_grad():
                    self._train_step(train_gen, epoch, self.optimizer, iterations=iterations)
            else:
                self._train_step(train_gen, epoch, self.optimizer, iterations=iterations)

            torch.cuda.empty_cache()
            top1_acc = self.test(test_gen)
            if top1_acc > self.best_top1_acc:
                self.best_top1_acc = top1_acc
                self._save_state(epoch, desc=desc)
        return top1_acc

    def _train_step(self, train_gen, epoch, optimizer, iterations=None, bn_train=True):
        self.model.train()

        # BN to gather running mean and var (does not require backprop!)
        if not bn_train:
            for m in self.model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

        batch_time = self.AverageMeter('Time', ':6.3f')
        data_time = self.AverageMeter('Data', ':6.3f')
        losses = self.AverageMeter('Loss', ':.4e')
        progress = self.ProgressMeter(len(train_gen), batch_time, data_time, losses, prefix="Epoch: [{}]".format(epoch))

        end = time.time()
        for i, (input, target) in enumerate(train_gen):
            # measure data loading time
            data_time.update(time.time() - end)
            input = input.cuda(self.device, non_blocking=True)
            target = target.cuda(self.device, non_blocking=True)
            # new_shape = [1] + list(input.size())
            # input = torch.reshape(input, new_shape)
            # print(input.shape)

            # Compute output
            output = self.parallel_model(input)
            output = torch.flatten(output, 1)
            loss = self.criterion(output, target)

            # Record loss
            losses.update(loss.item(), input.size(0))

            # Compute gradient and do SGD step
            # Bypass when the learning rate is zero
            if self.optimizer.param_groups[0]['lr'] != 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.print(i)

            if iterations is not None:
                if i == iterations:
                    break

    def _adjust_lr_rate(self, optimizer, epoch, lr_list, gamma):
        if lr_list is None:
            return
        for value in lr_list:
            if epoch == value:
                new_lr = self.optimizer.param_groups[0]['lr'] * gamma
                cfg.LOG.write("=> New learning rate set to {}".format(self.optimizer.param_groups[0]['lr'] * gamma))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                    print('my new lr is: {}'.format(new_lr))

    def _save_state(self, epoch, desc=None):
        if desc is None:
            filename = '{}_epoch-{}_top1-{}.pth'.format(self.arch, epoch, round(self.best_top1_acc, 2))
        else:
            filename = '{}_epoch-{}_{}_top1-{}.pth'.format(self.arch, epoch, desc, round(self.best_top1_acc, 2))
        path = '{}/{}'.format(cfg.LOG.path, filename)

        state = {'arch': self.arch,
                 'epoch': epoch + 1,
                 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'lr_plan': self.lr_plan,
                 'best_top1_acc': self.best_top1_acc}

        torch.save(state, path)

    def _load_state(self, path):
        if os.path.isfile(path):
            chkp = torch.load(path)

            # Load class variables from checkpoint
            self.next_train_epoch = chkp['epoch'] if 'epoch' in chkp else 0

            # Assuming parameters are either in state_dict or in model keys
            state_dict = chkp['state_dict'] if 'state_dict' in chkp else chkp['model']
            try:
                self.model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                cfg.LOG.write('Loading model state warning, please review')
                cfg.LOG.write('{}'.format(e))

            self.optimizer_state = chkp['optimizer']
            self.lr_plan = chkp['lr_plan'] if 'lr_plan' in chkp else None
            self.best_top1_acc = chkp['best_top1_acc'] if 'best_top1_acc' in chkp else 0

            cfg.LOG.write("Checkpoint best top1 accuracy is {} @ epoch {}"
                          .format(round(self.best_top1_acc, 2), self.next_train_epoch - 1))
        else:
            cfg.LOG.write("Unable to load model checkpoint from {}".format(path))
            raise RuntimeError

    def print_stats(self, only_cfg=False, t1_analysis=False):
        raise NotImplementedError

    @staticmethod
    def _accuracy(output, target, topk=(1,)):
        with torch.no_grad():
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

    @staticmethod
    class AverageMeter(object):
        def __init__(self, name, fmt=':f'):
            self.name = name
            self.fmt = fmt
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

        def __str__(self):
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
            return fmtstr.format(**self.__dict__)

    @staticmethod
    class ProgressMeter(object):
        def __init__(self, num_batches, *meters, prefix=""):
            self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
            self.meters = meters
            self.prefix = prefix

        def print(self, batch):
            entries = [self.prefix + self.batch_fmtstr.format(batch)]
            entries += [str(meter) for meter in self.meters]
            cfg.LOG.write('\t'.join(entries))

        def _get_batch_fmtstr(self, num_batches):
            num_digits = len(str(num_batches // 1))
            fmt = '{:' + str(num_digits) + 'd}'
            return '[' + fmt + '/' + fmt.format(num_batches) + ']'
