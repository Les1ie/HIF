import torch
from torch import Tensor, nn
from torch.nn import init
from torch.nn.utils import weight_norm

from nn.functional import nl_unfold, nl_fold
from nn.functional.dilation_func import *


class NonLinearConv1d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation_func=None, dilation_kwargs={}) -> None:
        super(NonLinearConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._kernel_size = (kernel_size, in_channels)

        # self.weights = nn.Parameter(torch.Tensor(kernel_size, in_channels, out_channels))
        # self.bias = nn.Parameter(torch.Tensor(kernel_size, in_channels, out_channels))
        self.weight = nn.Parameter(torch.Tensor((kernel_size * in_channels), out_channels))
        self.bias = nn.Parameter(torch.Tensor(1, out_channels))

        self._dilation_func = dilation_func
        self._dilation = None
        self.check_apply_dilation_func(dilation_func, dilation_kwargs, kernel_size - 1)
        self.init_weights()
        # self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

    def check_apply_dilation_func(self, dilation_func, dilation_kwargs, kernel_size):
        if dilation_func in ['fib', 'sin', 'exp']:
            dilation_map = {
                'fib': fib_dilation,
                'sin': sin_dilation,
                'exp': exp_dilation
            }
            self._dilation = dilation_map[dilation_func](kernel_size, **dilation_kwargs)
        elif hasattr(dilation_func, '__call__'):
            self._dilation = dilation_func(kernel_size, **dilation_kwargs)
        else:
            raise Exception('Expected parameter dilation_func must in ["fib", "sin", "exp"] or be a function, '
                            'unknown type "{}".'.format(type(dilation_func).__name__))

    def init_weights(self):
        init.kaiming_uniform_(self.weight)
        init.kaiming_uniform_(self.bias)

    @property
    def kernel_size(self) -> int:
        return self._kernel_size

    @property
    def dilation(self):
        return self._dilation

    def forward(self, input: Tensor) -> Tensor:
        b, l, d = input.shape
        # unfolds = F.unfold(input, self.kernel_size)
        unfolds = nl_unfold(input, self.kernel_size, dilation=self.dilation)
        x = torch.einsum('bdg,do->bog', unfolds, self.weight)
        x = x + self.bias.T
        # folds = F.fold(x, output_size=(l - self.kernel_size[0] + 1, self.out_channels),
        #                kernel_size=(1, self.out_channels))
        folds = nl_fold(x, output_size=(l - self.kernel_size[0] + 1, self.out_channels),
                        kernel_size=(1, self.out_channels))
        # output = torch.squeeze(folds, 1)
        return folds


class NonLinearTemporalConvNet(nn.Module):
    def __init__(self, in_feats, hid_feats, kernel_size=4, num_layers=3, dropout=0.1, dilation_func='sin'):
        super(NonLinearTemporalConvNet, self).__init__()
        self.layers = []
        for i in range(num_layers):
            ipt = in_feats if i == 0 else hid_feats
            opt = hid_feats
            layer = [
                weight_norm(NonLinearConv1d(ipt, opt, kernel_size, dilation_func=dilation_func)),
                # nn.LayerNorm(opt),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.ZeroPad2d((0, 0, 0, kernel_size - 1)),
            ]
            self.layers.extend(layer)
        self.network = nn.Sequential(*self.layers)

    def forward(self, input):
        output = self.network(input)
        return output


if __name__ == '__main__':
    bs = 1
    sl = 10
    dim = 4
    input = torch.randn((bs, sl, dim))
    # conv = NonLinearConv1d(dim, 3, kernel_size=2, dilation_func='sin')
    conv = NonLinearTemporalConvNet(dim, 3, kernel_size=2,
                                    num_layers=2,
                                    dilation_func='sin')
    # print('parameters:', list(conv.parameters()))
    # print('dilation:', conv.dilation)
    # print('weights:', conv.weights)
    # print('bias:', conv.bias)
    output = conv(input)
    print(output)
