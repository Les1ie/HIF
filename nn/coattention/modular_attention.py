from copy import deepcopy

import torch
from torch import nn

__all__ = ['ModularAttentionUnit', 'SelfGuidedAttentionLayer', 'CoAttentionLayer', 'CoAttentionEncoderDecoder']

from torch.nn import init

'''
paper: Deep Modular Co-Attention Networks for Visual Question Answering, CVPR 2019
'''


class ModularAttentionUnit(nn.Module):
    '''
            input
              ├─────────┐
              ▼         │
          Attention     │
              │         │
              ▼         │
        Add&LayerNorm ◄─┘
              ├─────────┐
              ▼         │
         Feedforward    │
              │         │
              ▼         │
        Add&LayerNorm ◄─┘
              │
              ▼
            output
    '''

    def __init__(self, feats, num_heads, dropout, activate=nn.LeakyReLU):
        super(ModularAttentionUnit, self).__init__()
        self.feats = feats
        hid_feats = 4 * feats
        self.hid_feats = hid_feats
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention = nn.MultiheadAttention(feats, num_heads=num_heads, batch_first=True, bias=False,
                                               dropout=dropout)
        self.mlps = nn.Sequential(
            nn.Linear(feats, hid_feats),
            activate(),
            nn.Dropout(dropout),
            nn.Linear(hid_feats, feats),
        )
        self.layer_norm = nn.LayerNorm(feats)
        self.layer_norm_out = nn.LayerNorm(feats)

    def forward(self, q, k=None, v=None, **kwargs):
        '''
        forward(X): Self-Attention takes one group of input features X and output the attended features for X
        forward(X, Y): Guided-Attention takes two groups of input features X and Y and output the attended features
          for X guided by Y.
        '''
        is_self_attention = (k is None and v is None)
        if k is None:
            k = q
        if v is None:
            v = k
        # 检查参数
        attn_mask = kwargs.get('attn_mask', None)
        if attn_mask is not None:
            sp = attn_mask.shape
            if len(sp) == 3 and sp[0] != q.size(0) * self.num_heads:
                del kwargs['attn_mask']
            if sp[-2] != q.size(1) or sp[-1] != k.size(1):
                del kwargs['attn_mask']
        values, weights = self.attention(q, k, v, **kwargs)
        nan_fill = 0
        values = torch.masked_fill(values, torch.isnan(values), nan_fill)
        weights = torch.masked_fill(weights, torch.isnan(weights), nan_fill)
        x = q + values
        x = self.layer_norm(x)
        x = x + self.mlps(x)
        x = self.layer_norm_out(x)
        if is_self_attention and 'key_padding_mask' in kwargs.keys():
            x[kwargs['key_padding_mask']] = 0
        return x


class SelfGuidedAttentionLayer(nn.Module):
    '''
    SGA(X,Y) = GA(SA(X), Y)
    '''

    def __init__(self, feats, num_heads=4, dropout=0.3, activate=nn.LeakyReLU):
        super(SelfGuidedAttentionLayer, self).__init__()
        self.feats = feats
        hid_feats = 4 * feats
        self.hid_feats = hid_feats
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.sa_x = ModularAttentionUnit(feats=feats, num_heads=num_heads, dropout=dropout, activate=activate)
        self.ga_xy = ModularAttentionUnit(feats=feats, num_heads=num_heads, dropout=dropout, activate=activate)

    def forward(self, X, Y, **kwargs):
        return self.ga_xy(self.dropout(self.sa_x(X, **kwargs)), Y, **kwargs)


class CoAttentionLayer(SelfGuidedAttentionLayer):
    '''
    CA(X, Y) = SGA(X, SA(Y)) = GA(SA(X), SA(Y))
    '''

    def __init__(self, feats, num_heads=4, dropout=0.3, activate=nn.LeakyReLU):
        super(CoAttentionLayer, self).__init__(feats=feats, num_heads=num_heads, dropout=dropout, activate=activate)
        self.sa_y = ModularAttentionUnit(feats=feats, num_heads=num_heads, dropout=dropout, activate=activate)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, Y, **kwargs):
        sa_x = self.dropout(self.sa_x(X, **kwargs))
        sa_y = self.dropout(self.sa_y(Y, **kwargs))
        return self.ga_xy(sa_x, sa_y)


class StackedCoAttention(nn.Module):
    def __init__(self, feats, num_layers=1, num_heads=4, dropout=0.3, activate=nn.LeakyReLU):
        super(StackedCoAttention, self).__init__()
        self.num_layers = num_layers
        self.ca_layers = nn.ModuleList([CoAttentionLayer(feats, num_heads, dropout, activate)
                                        for i in num_layers])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, **kwargs):
        for ca in self.ca_layers:
            x, y = ca(X=x, Y=y, **kwargs)
            x = self.dropout(x)
            y = self.dropout(y)
        return x, y


class CoAttentionEncoderDecoder(nn.Module):
    def __init__(self, feats, num_layers=3, num_heads=4, dropout=0.3, activate=nn.LeakyReLU):
        super(CoAttentionEncoderDecoder, self).__init__()
        self.feats = feats
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.activate = activate
        self.encoder_sa_layers = nn.ModuleList([ModularAttentionUnit(feats=feats, num_heads=num_heads,
                                                                     dropout=dropout, activate=activate)
                                                for i in range(num_layers)])
        self.decoder_ga_layers = nn.ModuleList([ModularAttentionUnit(feats=feats, num_heads=num_heads,
                                                                     dropout=dropout, activate=activate)
                                                for i in range(num_layers)])
        self.decoder_sa_layers = nn.ModuleList([ModularAttentionUnit(feats=feats, num_heads=num_heads,
                                                                     dropout=dropout, activate=activate)
                                                for i in range(num_layers)])

    def forward(self, X, Y, **kwargs):
        '''
        kwargs: attn_mask_x=None, attn_mask_y=None, attn_mask_xy=None
        '''
        kwargs_x = {k[:-2]: v for k, v in kwargs.items() if k.endswith('_x')}
        kwargs_y = {k[:-2]: v for k, v in kwargs.items() if k.endswith('_y')}
        kwargs_xy = {k[:-3]: v for k, v in kwargs.items() if k.endswith('_xy')}
        for i in range(self.num_layers):
            encoder = self.encoder_sa_layers[i]
            Y = encoder(Y, **kwargs_y)
            # if i > 0:
            #     Y = self.dropout(Y)
        for i in range(self.num_layers):
            decoder_sa = self.decoder_sa_layers[i]
            decoder_ga = self.decoder_ga_layers[i]
            X = decoder_sa(X, **kwargs_x)
            X = decoder_ga(X, Y, **kwargs_xy)
            # if i > 0:
            #     X = self.dropout(X)
        return X, Y


if __name__ == '__main__':
    dim = 4
    x = torch.randn([1, 3, dim])
    y = torch.randn([1, 6, dim])
    ca = CoAttentionEncoderDecoder(dim)
    rst = ca(x, y)
    print(x)
    print(y)
    print(rst)
