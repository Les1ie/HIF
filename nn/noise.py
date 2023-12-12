from typing import Optional
import numpy as np
import torch
from torch import nn

__all__ = ['Noise', 'CrossNoise']


class Noise(nn.Module):
    def __init__(self, in_feats, noise_weight: Optional[float] = 0.1, noise_rate=1, noise_dim=0, *args, **kwargs):
        '''
        todo 增加固定值噪音功能
        Simply add noise into input tensor.
        @param noise_weight: weights of noise tensor
        @param rate: rows rate to add noise
        @param noise_dim: dimension to add noise, 0 on rows, 1 on columns
        '''
        super(Noise, self).__init__()
        self.in_feats = in_feats
        self.dim = noise_dim
        self.alpha = noise_weight
        self.rate = noise_rate
        self.dims = None
        if self.dim == 1:
            # 按列添加噪音时，固定使用最后 num_cols 列
            num_cols = int(self.rate * self.in_feats)
            self.dims = list(range(self.in_feats - num_cols, self.in_feats))

    def forward(self, input):
        if not self.alpha or not self.rate:
            # skip noise
            return input
        input_shape = num_rows, num_col = input.size()[-2:]
        cnt = int(self.rate * input_shape[self.dim])
        noise_shape = [num_rows, num_col]
        noise_shape[self.dim] = cnt
        noise = torch.normal(0, 1, noise_shape, device=input.device)
        idx = np.random.choice(np.arange(0, input_shape[self.dim]), cnt, False)
        if self.dim == 0:
            input[idx] += (1 - self.alpha) * input[idx] + noise * self.alpha
        else:
            input[:, idx] += (1 - self.alpha) * input[:, idx] + noise * self.alpha
        return input


class CrossNoise(nn.Module):
    def __init__(self, scope, **kwargs):
        '''

        @param scope: select from "batch", "cascade" or "subcascade"
        @param kwargs:
        '''
        self.scope = scope
        super(CrossNoise, self).__init__(**kwargs)

    def forward(self):
        pass


if __name__ == '__main__':
    noise = Noise(0.2, 0.4)
    test_tensor = torch.randn([5, 4])
    print(test_tensor)
    out = noise(test_tensor)
    print(out)
    pass
