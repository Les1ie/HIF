import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TextCNN(nn.Module):
    '''
    https://github.com/649453932/Chinese-Text-Classification-Pytorch/blob/master/models/TextCNN.py
    '''
    def __init__(self, dataset, embedding, vocab_path):
        super(TextCNN, self).__init__()
        self.model_name = 'TextCNN'
        self.vocab_path = vocab_path                                    # 词表

        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None                          # 预训练词向量

        self.dropout = 0.5                                              # 随机失活
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)

        if self.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(self.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(self.n_vocab, self.embed, padding_idx=self.n_vocab - 1)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.embed)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(self.dropout)
        # self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        # out = self.fc(out)
        return out

if __name__ == '__main__':
    os.getcwd()
    model = TextCNN()
