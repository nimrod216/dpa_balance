import torch
from torch.nn.modules.module import Module
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction

class InfLoss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(InfLoss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

    def forward(self, input, target):
        return torch.max(torch.abs(input-target))

