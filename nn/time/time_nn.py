__all__ = ['TimeRNN', 'TimeTransformer']

from torch import nn
from torch.nn import functional as F
from .base_module import BaseTimeModule



class TimeRNN(BaseTimeModule):
    def __init__(self, rnn='gru', num_rnn_layers=1,
                 dropout=0,
                 bidirectional=True,
                 norm=True,
                 **kwargs):
        super(TimeRNN, self).__init__(**kwargs)

        self.rnn_name = rnn
        self.num_rnn_layers = num_rnn_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.norm = norm
        self.layer_norm = nn.LayerNorm([self.num_time_nodes, self.in_feats]) if norm else None
        rnn_cls = None
        if rnn == 'gru':
            rnn_cls = nn.GRU
        elif rnn == 'lstm':
            rnn_cls = nn.LSTM
        else:
            raise Exception(f"Invalid rnn class: {rnn}")

        self.rnn = rnn_cls(**self.rnn_factory)
        self.mlp = nn.Linear(self.in_feats * (self.bidirectional + 1),
                             self.in_feats)

    @property
    def rnn_factory(self):
        args = {
            'input_size': self.in_feats,
            'hidden_size': self.in_feats,
            'num_layers': self.num_rnn_layers,
            'batch_first': True,
            'bidirectional': self.bidirectional,
            'dropout': self.dropout,
        }
        return args

    def forward(self, time_x, **kwargs):
        self.rnn.flatten_parameters()
        time_x = self.unbatch(time_x)
        time_x, hidden_states = self.rnn(time_x)
        time_x = self.mlp(time_x)
        if self.norm:
            time_x = self.layer_norm(time_x)
        time_x = self.batch(time_x)
        return time_x


class TimeTransformer(BaseTimeModule):

    def __init__(self, num_encoder_layers=1,
                 dropout=0,
                 **kwargs):
        super(TimeTransformer, self).__init__(**kwargs)
        self.num_heads = kwargs.get('num_heads', 4)
        self.dropout = dropout
        self.num_encoder_layers = num_encoder_layers
        encoder_layer = nn.TransformerEncoderLayer(**self.encoder_layer_factory)
        encoder_factory = {
            'encoder_layer': encoder_layer,
            'num_layers': self.num_encoder_layers,
            'norm': nn.LayerNorm([self.num_time_nodes, self.in_feats]),
        }
        self.transformer_encoder = nn.TransformerEncoder(**encoder_factory)
        from .. import NonparametricPositionalEncoding
        self.pe = NonparametricPositionalEncoding(self.in_feats, dropout=self.dropout)

    @property
    def encoder_layer_factory(self):
        args = {
            'd_model': self.in_feats,
            'nhead': self.num_heads,
            'batch_first': True,
        }
        return args

    def forward(self, time_x, **kwargs):
        time_x = self.unbatch(time_x)
        time_x = self.transformer_encoder(time_x)
        time_x = self.batch(time_x)
        return time_x
