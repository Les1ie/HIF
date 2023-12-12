import math

import torch
from torch import nn


class NonparametricPositionalEncoding(nn.Module):
    def __init__(self, in_feats, dropout=0.1, max_len=512):
        super(NonparametricPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, in_feats)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, in_feats, 2).float() * (-math.log(10000.0) / in_feats))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TimeAwarePositionalEncoding(nn.Module):
    def __init__(self, in_feats, dropout=0.1, max_len=512, agg='atten', batch_first=True, num_heads=4):
        super(TimeAwarePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.in_feats = in_feats
        self.agg = agg
        self.batch_first = batch_first
        self.time_mlp = nn.Sequential(
            nn.Linear(1, in_feats),
            nn.Dropout(p=dropout),
            nn.ELU(),
            nn.Linear(in_feats, in_feats),
        )
        self.pe_div_term = torch.exp(torch.arange(0, in_feats, 2).float() * (-math.log(10000.0) / in_feats))
        self.time_div_term = torch.exp(torch.arange(0, in_feats, 2).float() * (-math.log(10000.0) / in_feats))
        self.position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        pe = torch.zeros(max_len, in_feats)
        pe[:, 0::2] = torch.sin(self.position * self.pe_div_term)
        pe[:, 1::2] = torch.cos(self.position * self.pe_div_term)
        pe = pe.unsqueeze(0)
        if not batch_first:
            pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)
        self.atten = None
        if agg not in ['atten', 'sum']:
            raise NotImplementedError(f'Unknown aggregate function: "{agg}".')
        if agg == 'atten':
            self.atten = nn.MultiheadAttention(in_feats, num_heads=num_heads, dropout=dropout, batch_first=batch_first)

    def forward(self, x, seq_time):
        time_pe = torch.zeros_like(x)
        time_pe[:, :, 0::2] = torch.sin((seq_time) * self.time_div_term)
        time_pe[:, :, 1::2] = torch.sin((seq_time) * self.time_div_term)
        if self.agg == 'atten':
            if self.batch_first:
                pe = self.pe.repeat((x.size(0), 1, 1))
                truncated_pe = pe[:, :x.size(1), :]
            else:
                truncated_pe = self.pe[:x.size(1), :, :]
            time_aware_pe, weights = self.atten(time_pe, truncated_pe, truncated_pe)
        else:
            if self.batch_first:
                truncated_pe = self.pe[:, :x.size(1), :]
            else:
                truncated_pe = self.pe[:x.size(1), :, :]
            time_aware_pe = time_pe + truncated_pe
        # time_pe = self.time_pe[:x.size(0), :]
        # seq_time = self.time_mlp(seq_time)
        x = x + time_aware_pe
        # seq_time = seq_time.repeat((1, 1, self.in_feats))
        # x = x + seq_time
        return self.dropout(x)


if __name__ == '__main__':
    d = 4
    max_len = 2
    bs = 3
    seq = torch.randn((bs, max_len, d))
    seq_time = torch.randn((bs, max_len, 1))
    print(seq)
    pe = TimeAwarePositionalEncoding(d, max_len=max_len * 2 ,agg='sum')
    print(pe(seq, seq_time))
