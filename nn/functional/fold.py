from itertools import accumulate
from math import prod

import torch
from torch.nn import functional as F

__all__ = ['nl_fold', 'nl_unfold']


def nl_fold(input, output_size, kernel_size, dilation=None, padding=0, stride=1):
    n, ck, l = input.shape
    assert ck % prod(kernel_size) == 0
    c = ck // prod(kernel_size)
    sp = [n]
    sp.extend(output_size)
    x = input.transpose(-1, -2).reshape(sp)
    return x


def nl_unfold(input, kernel_size, dilation, padding=0, stride=1):
    ignore_rows = kernel_size[0] - 1
    row_idx = [[i + j for j in accumulate([0] + dilation.tolist())] for i in range(input.size(1) - ignore_rows)]
    unfold = F.pad(input, [0, 0, 0, sum(dilation) - ignore_rows])[:, row_idx, :]
    shape = list(unfold.shape)
    unfold = unfold.view(shape[0], shape[1], prod(kernel_size)).transpose(-1, -2)
    return unfold


if __name__ == '__main__':
    t = torch.arange(12).view(1, 1, 4, 3).float()
    print('input:', t)
    o = nl_unfold(t, (3, 3), [1, 2])
    print('unfold:', o)
    kernel = torch.randn((3 * 3, 4)).float()
    print('kernel:', kernel)
    conv = torch.einsum('bkg,ko->bog', o, kernel)
    print('conv:', conv)
    f = nl_fold(conv, (3, 4), (1, 4))
    print('fold:', f)
