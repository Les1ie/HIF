import torch

__all__ = ['sequence_padding_collate', 'RandomEmbedCollater']

from torch_geometric.loader.dataloader import Collater

from utils import AdditiveDict


def sequence_padding_collate(batch):
    '''
    将batch中的多个序列数据填充至等长
    :param batch:
    :return:
    '''
    max_len = max(map(len, map(lambda x: x[0], batch)))
    labels = []
    sequence = []
    for seq, label in batch:
        labels.append([label])
        t = torch.tensor(seq, dtype=torch.int)
        seq_len = len(seq)
        if seq_len < max_len:
            zeros = torch.zeros((max_len - seq_len, 3), dtype=torch.int)
            t = torch.cat([t, zeros], 0)
        sequence.append(t)
    sequence = torch.stack(sequence, 0)
    labels = torch.tensor(labels, dtype=torch.float)
    return labels, sequence


class RandomEmbedCollater(Collater):
    '''
    在 torch_geometric.loader.dataloader.Collater 的基础上，进一步实现点边特征随机生成
    '''

    def __init__(self, dim, node_embed=True, edge_embed=False,
                 follow_batch=None, exclude_keys=None,
                 *args, **kwargs):
        super().__init__(follow_batch=follow_batch, exclude_keys=exclude_keys)
        self.dim = dim
        self.node_embed = node_embed
        self.edge_embed = edge_embed

    def __call__(self, batch):
        batch = super(RandomEmbedCollater, self).__call__(batch)
        x = {'node': None, 'edge': None}

        if self.node_embed:
            x['node'] = AdditiveDict()
            for ntype, num in batch.num_nodes_dict.items():
                x['node'][ntype] = torch.normal(0, 1, [num, self.dim])

        if self.edge_embed:
            x['edge'] = AdditiveDict()
            for etype, num in batch.num_edges_dict.items():
                x['edge'][etype] = torch.normal(0, 1, [num, self.dim])
        batch['x'] = x
        return batch
