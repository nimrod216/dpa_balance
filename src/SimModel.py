import torch
import torch.nn as nn
from tabulate import tabulate
import Config as cfg
from QuantConv2d import UnfoldConv2d

class SimModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.unfold_list = []  # All UnfoldConv2d layers


    def forward(self, x):
        raise NotImplementedError

    def update_unfold_list(self):
        self.apply(self._apply_unfold_list)

    def _apply_unfold_list(self, m):
        if type(m) == UnfoldConv2d:
            self.unfold_list.append(m)


    def set_sparq(self, v_x, v_w):
        for l in self.unfold_list:
            l._sparq_x = v_x
            l._sparq_w = v_w

    def set_quantize(self, v):
        for l in self.unfold_list + self.unfold_bn_list:
            l._quantize = v

    def set_quantization_bits(self, quantization):
        for l in self.unfold_list:
            l._q_bits = quantization

    def set_min_max_update(self, v):
        for l in self.unfold_list + self.unfold_bn_list:
            l._disable_min_max_update = not v

    def set_round(self, v):
        for l in self.unfold_list:
            l._is_round = v

    def set_shift_opt(self, v):
        for l in self.unfold_list:
            l._shift_opt = v

    def set_bit_group(self, v_x, v_w):
        for l in self.unfold_list:
            l._bit_group_x = v_x
            l._bit_group_w = v_w

    def set_group_sz(self, v_x, v_w):
        for l in self.unfold_list:
            l._group_sz_x = v_x
            l._group_sz_w = v_w

    def set_quantization_type(self, fp):
        for l in self.unfold_list + self.unfold_bn_list:
            l.fp = fp

    def set_per_filter_quant(self, per_filter):
        for l in self.unfold_list:
            l.filter_wise = per_filter

    def set_shift_bits(self, shift_bit):
        for l in self.unfold_list + self.unfold_bn_list:
            l.shift_bits = shift_bit

    def print_config(self):
        headers = None
        table = []

        for l in self.unfold_list:
            headers, config = l.get_status_arr()
            table.append(config)

        headers.insert(0, '#')
        cfg.LOG.write(tabulate(table, headers=headers, showindex=True), date=False)
