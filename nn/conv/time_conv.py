from typing import Dict, Optional, Union

import torch
from torch import Tensor, Size
from torch import nn
from torch.nn import functional as F
from torch.nn.init import kaiming_uniform_, kaiming_normal_
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import MessagePassing, HeteroConv
from torch_geometric.nn.dense import Linear
from torch_geometric.utils import softmax, dropout_adj

__all__ = ['HTSAConv']

from torch_geometric.nn.conv.han_conv import group

from torch_geometric.nn.inits import reset, glorot

from torch_geometric.typing import Adj, NodeType, EdgeType, Metadata


class HTSAConv(MessagePassing):
    def __init__(
            self,
            in_feats: Union[int, Dict[str, int]],
            out_feats: int,
            metadata: Metadata,
            edges_with_ts=[],
            num_heads: int = 1,
            negative_slope=0.2,
            dropout: float = 0.0,
            dropout_edge: float = 0.0,
            **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if not isinstance(in_feats, dict):
            in_feats = {node_type: in_feats for node_type in metadata[0]}

        self.num_heads = num_heads
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.negative_slope = negative_slope
        self.metadata = metadata
        self.edges_with_ts = edges_with_ts
        self.dropout = dropout
        self.dropout_edge = dropout_edge
        self.k_lin = Linear(out_feats, out_feats)
        self.q = nn.Parameter(torch.Tensor(1, out_feats))

        self.proj = nn.ModuleDict()
        for node_type, in_feats in self.in_feats.items():
            self.proj[node_type] = nn.Linear(in_feats, out_feats)

        self.proj_e = nn.ModuleDict()
        self.lin_src = nn.ParameterDict()
        self.lin_dst = nn.ParameterDict()
        self.lin_e = nn.ParameterDict()
        self.lin_ts = nn.ParameterDict()
        self.lin_agg_e = nn.ModuleDict()
        dim = out_feats // num_heads
        for edge_type in metadata[1]:
            with_timestamp = edge_type in self.edges_with_ts
            edge_type = '__'.join(edge_type)
            self.lin_src[edge_type] = nn.Parameter(torch.Tensor(1, num_heads, dim))
            self.lin_dst[edge_type] = nn.Parameter(torch.Tensor(1, num_heads, dim))
            self.lin_e[edge_type] = nn.Parameter(torch.Tensor(1, num_heads, dim * (2 + with_timestamp)))
            self.lin_ts[edge_type] = nn.Parameter(torch.Tensor(1, out_feats))
            self.proj_e[edge_type] = nn.Linear(in_feats, out_feats)
            self.lin_agg_e[edge_type] = nn.Linear((3 + with_timestamp) * out_feats, out_feats)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.proj)
        reset(self.proj_e)
        reset(self.lin_agg_e)
        for edge_type in self.metadata[1]:
            with_timestamp = edge_type in self.edges_with_ts
            edge_type = '__'.join(edge_type)
            kaiming_normal_(self.lin_src[edge_type])
            kaiming_normal_(self.lin_e[edge_type])
            kaiming_normal_(self.lin_dst[edge_type])
            kaiming_normal_(self.lin_ts[edge_type])
        kaiming_normal_(self.q)
        self.k_lin.reset_parameters()

    def forward(self, x_dict: Dict[NodeType, Tensor],
                edge_index_dict: Dict[EdgeType, Adj],
                edge_x_dict: Dict[EdgeType, Tensor],
                timestamp_dict: Dict[EdgeType, Tensor],
                ) -> Dict[NodeType, Optional[Tensor]]:
        H, D = self.num_heads, self.out_feats // self.num_heads
        x_node_dict, out_dict = {}, {}
        edge_out_dict = {} if edge_x_dict else None
        out_edge_index_dict = {}
        # Iterate over node types:
        for node_type, x_node in x_dict.items():
            x_node_dict[node_type] = self.proj[node_type](x_node).view(
                -1, H, D)
            out_dict[node_type] = []

        # Iterate over edge types:
        for edge_type, edge_index in edge_index_dict.items():
            meta_edge = edge_type
            src_type, _, dst_type = meta_edge
            edge_type = '__'.join(edge_type)
            x_dst = x_node_dict[dst_type]
            x_src = x_node_dict[src_type]
            lin_src = self.lin_src[edge_type]
            lin_dst = self.lin_dst[edge_type]
            lin_e = self.lin_e[edge_type]
            # todo: 使用 边类型+点类型 获取参数，起到同类型点在不同边中有不同作用的效果
            alpha_src = (x_src * lin_src).sum(dim=-1)
            alpha_dst = (x_dst * lin_dst).sum(dim=-1)
            if edge_x_dict:
                x_edge = self.proj_e[edge_type](edge_x_dict[meta_edge]).view(-1, H, D)
                if meta_edge in self.edges_with_ts:
                    ts = timestamp_dict[meta_edge]
                    lin_ts = self.lin_ts[edge_type]
                    x_ts = (ts * lin_ts).view(-1, H, D)
                    x_e = (torch.cat([x_edge, x_ts], -1))
                else:
                    x_e = x_edge
                if 'user' in (src_type, dst_type) and self.dropout_edge != 0:
                    edge_index, x_e = dropout_adj(edge_index, x_e, self.dropout_edge, training=self.training)
                alpha_e = (torch.cat([x_e, x_dst[edge_index[1]]], -1) * lin_e).sum(dim=-1)
                # 针对 用户-时间 边的软划分，以用户为 index 进行 softmax
                if 'user' in (src_type, dst_type) and 'time' in (dst_type, src_type):
                    if 'user' == src_type:
                        alpha_e = softmax(alpha_e, edge_index[0])
                    else:
                        alpha_e = softmax(alpha_e, edge_index[1])

            else:
                if 'user' in (src_type, dst_type) and self.dropout_edge != 0:
                    edge_index, _ = dropout_adj(edge_index, self.dropout_edge, training=self.training)

                alpha_e = None
            alpha = (alpha_src, alpha_dst)
            # propagate_type: (x_dst: Tensor, alpha: PairTensor)
            out = self.propagate(edge_index, x=(x_src, x_dst), alpha=alpha,
                                 x_e=x_e, alpha_e=alpha_e, size=None)
            # out = self.update(out, )
            # out = F.leaky_relu(out, self.negative_slope)
            out_dict[dst_type].append(out)
            if edge_x_dict:
                edge_out = self.edge_updater(edge_index=edge_index, edge_type=edge_type,
                                             x_edge=x_e.view(x_e.size(0), -1),
                                             x=(x_src.view(-1, self.out_feats), out))
                # edge_out_dict[meta_edge] = F.leaky_relu(edge_out, self.negative_slope)
                edge_out_dict[meta_edge] = edge_out
            out_edge_index_dict[tuple(edge_type.split('__'))] = edge_index

        # iterate over node types:
        for node_type, outs in out_dict.items():
            out = group(outs, self.q, self.k_lin)

            if out is None:
                out_dict[node_type] = None
                continue
            out_dict[node_type] = out

        return {'node': out_dict, 'edge': edge_out_dict}, out_edge_index_dict

    def message(self, x_i: Tensor, x_j: Tensor, alpha_i: Tensor, alpha_j: Tensor, alpha_e,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        if alpha_e is not None:
            alpha = (alpha_i + alpha_j) * alpha_e
        else:
            alpha = alpha_i + alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        # 对每一条入边进行softmax，由于信息传播级联中点的入度恒定为1，所以导致节点的 alpha=1
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = x_j * alpha.view(-1, self.num_heads, 1)
        return out.view(-1, self.out_feats)

    def edge_update(self, edge_index, edge_type, x_edge, x_i, x_j) -> Tensor:
        x_edge = torch.cat([x_j, x_edge, x_i], -1)
        lin = self.lin_agg_e[edge_type]
        out = lin(x_edge)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_feats}, '
                f'num_heads={self.num_heads})')


if __name__ == '__main__':
    data = HeteroData()
    dim = 4
    uru = 'user', 'repost', 'user'
    ufu = 'user', 'follow', 'user'
    upt = 'user', 'postat', 'time'
    data['user'].x = torch.randn([3, dim], dtype=torch.float)
    data['time'].x = torch.randn([2, dim], dtype=torch.float)
    data[uru].edge_index = torch.tensor([[0, 1, 1, 2],
                                         [1, 0, 2, 1]], dtype=torch.long)
    data[ufu].edge_index = torch.tensor([[0, 0, 1],
                                         [2, 1, 0]], dtype=torch.long)
    data[upt].edge_index = torch.tensor([[0, 0, 1, 1, 2, 2],
                                         [0, 1, 0, 1, 0, 1]], dtype=torch.long)
    data[uru].edge_x = torch.randn([4, dim])
    data[ufu].edge_x = torch.randn([3, dim])
    data[upt].edge_x = torch.randn([6, dim])
    data[uru].timestamp = torch.randn([4, 1])
    data[upt].timestamp = torch.randn([3, 1])
    # data[uru].x = torch.randn([4, dim])
    # data[ufu].x = torch.randn([3, dim])
    # conv = TSAConv(dim)
    hetero_conv = HTSAConv(in_feats=dim, out_feats=dim, metadata=data.metadata(),
                           edges_with_ts=[uru])
    x_dict, edge_index_dict = hetero_conv(x_dict=data.x_dict,
                         edge_x_dict=data.edge_x_dict,
                         edge_index_dict=data.edge_index_dict,
                         timestamp_dict=data.timestamp_dict
                         )
    print(x_dict['node'])
    # edge_x_dict = {tuple(k.split('__')): conv.edge_x for k, conv in hetero_conv.convs.items()}
    print(x_dict['edge'])
