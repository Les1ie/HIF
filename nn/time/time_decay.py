__all__ = ['TimeDecay']

import torch
from torch import nn
from torch.nn import init
from .base_module import BaseTimeModule


class TimeDecay(BaseTimeModule):

    def __init__(self, cumulate=False,
                 norm=False,
                 **kwargs):
        super(TimeDecay, self).__init__(**kwargs)
        self.cumulate = cumulate
        self.norm = norm

        self.time_decay_weights = nn.Parameter(torch.Tensor(1, self.num_time_nodes, self.in_feats))
        self.time_decay_cum_weights = nn.Parameter(torch.Tensor(1, self.num_time_nodes, self.in_feats))
        self.layer_norm = nn.LayerNorm([self.num_time_nodes, self.in_feats]) if norm else None
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.time_decay_weights)
        init.kaiming_normal_(self.time_decay_cum_weights)

    @property
    def hyperparameters(self):
        params = {
            'cumulate': self.cumulate,
        }
        return params

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        group = parser.add_argument_group('TimeDecay')
        group.add_argument('--cumulate', type=bool, default=False, help='accumulate the weights of time decay.')
        return parser

    def forward(self, time_x, **kwargs):
        time_x = self.unbatch(time_x)
        weights = self.time_decay_weights
        if self.cumulate:
            cum_weights = self.time_decay_cum_weights
            cumulated_weights = weights * cum_weights
            weights = torch.cumsum(cumulated_weights, 1)
        time_x = time_x * weights
        time_x = time_x + time_x
        if self.norm:
            time_x = self.layer_norm(time_x)
        time_x = self.batch(time_x)
        return time_x
