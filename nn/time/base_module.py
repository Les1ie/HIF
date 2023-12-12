from torch import nn

__all__ = ['BaseTimeModule']


class BaseTimeModule(nn.Module):
    def __init__(self, in_feats: int, num_time_nodes: int, **kwargs):
        self.in_feats = in_feats
        self.num_time_nodes = num_time_nodes
        super(BaseTimeModule, self).__init__()

    def forward(self, time_x, *args, **kwargs):
        return time_x

    def batch(self, time_x):
        batched = self.is_batched(time_x)
        if not batched:
            time_x = time_x.view(-1, self.in_feats)
        return time_x

    @staticmethod
    def is_batched(time_x):
        return time_x.dim() == 2

    def unbatch(self, time_x):
        batched = self.is_batched(time_x)
        if batched:
            time_x = time_x.view(-1, self.num_time_nodes, self.in_feats)
        return time_x


