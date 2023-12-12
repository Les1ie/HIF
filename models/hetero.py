from copy import deepcopy
from typing import Any, Optional

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import HeteroConv, GCNConv, HANConv, GraphNorm, bro, LayerNorm
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_scatter import scatter
from matplotlib import pyplot as plt
from os import path as osp
import numpy as np

from models import Predictor
from models.mixin import ConvMixin, ModuleEmbeddings, ModuleReadout, ModuleTimeEmbed
from nn import Noise, HTSAConv, TimeDecay, TimeRNN, CoAttentionReadout
from utils import AdditiveDict
from utils.draw import plot_result, collect_result


class HeteroGCNPopularityPredictor(
    ModuleEmbeddings,
    ModuleReadout,
    ModuleTimeEmbed,
    ConvMixin,
    Predictor,
):
    name = 'Hetero'

    def __init__(self,
                 meta_edges=None,
                 dropout_edge=0.2,
                 num_gcn_layers=2,
                 noise_weight=None, noise_rate=0, noise_dim=1,
                 add_self_loops=False,
                 *args, **kwargs):
        self.dropout_edge = dropout_edge
        self.meta_edges = meta_edges
        self.num_gcn_layers = num_gcn_layers
        self.add_self_loops = add_self_loops
        super().__init__(*args, **kwargs)

        self.time_emb_fuse_mlp = nn.Linear(2 * self.in_feats, self.in_feats)
        self.noise_weight = noise_weight
        self.noise_rate = noise_rate
        self.noise_dim = noise_dim
        self.noise = Noise(in_feats=self.in_feats, noise_weight=noise_weight, noise_rate=noise_rate,
                           noise_dim=noise_dim)
        self.save_hyperparameters('num_gcn_layers', 'total_nodes', 'noise_weight', 'noise_dim', 'noise_rate',
                                  'add_self_loops', 'dropout_edge',
                                  )
        # self.reset_parameters()

    def get_convs(self):
        convs = []
        for i in range(self.num_gcn_layers):
            in_channels = self.in_feats if i != 0 else self.in_feats
            out_channels = self.in_feats
            conv = HeteroConv({me: GCNConv(in_channels=in_channels, out_channels=out_channels)
                               for me in self.meta_edges}, aggr='sum')
            convs.append(conv)
        return convs

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        group = parser.add_argument_group('HeteroPredictor')
        # noise_weight=None, noise_rate=0, noise_dim=1,
        group.add_argument('--num_gcn_layers', type=int, default=2, help='the number of gcn layers.')
        group.add_argument('--dropout_edge', type=float, default=0, help='dropout edges.')
        group.add_argument('--add_self_loops', type=bool, default=True, help='whether add self-loops to graphs.')
        group.add_argument('--noise_weight', type=float, default=0, help='weight of noise.')
        group.add_argument('--noise_rate', type=float, default=0.1, help='coverage rate of noise.')
        group.add_argument('--noise_dim', type=int, default=1, choices=[0, 1], help='dimension of noise.')
        return parser

    def forward(self, batch) -> Any:
        # to do 增加自环会导致带时间戳的边特征生成错误
        x, edge_index_dict = self.check_self_loops(batch=batch)
        result = self.conv(x=x, batch=batch, edge_index_dict=edge_index_dict)
        x = result['x']
        batch = result['batch']
        edge_index_dict = result['edge_index_dict']
        return self.readout(x=x, batch=batch, edge_index_dict=edge_index_dict)

    def check_self_loops(self, batch):
        if self.add_self_loops:
            edge_index_dict = {}
            for k, v in batch.edge_index_dict.items():
                if k[0] == k[2]:
                    # 仅对同节点类型的关系添加自环
                    v, _ = remove_self_loops(v)
                    if v.size(-1) == 0:
                        # 仅包含自环边时，移除后就没其他边了v.size(-1)=0，会导致错误
                        v = batch.edge_index_dict[k]
                    else:
                        v, _ = add_self_loops(v)
                edge_index_dict[k] = v
        else:
            edge_index_dict = batch.edge_index_dict
        x = {'node': self.get_node_embeddings(batch),
             'edge': self.get_edge_embeddings(batch, edge_index_dict=edge_index_dict)}
        return x, edge_index_dict

    def metrics(self, stage, *args, **kwargs):
        metrics = super(HeteroGCNPopularityPredictor, self).metrics(stage, *args, **kwargs)
        metrics = self.readout.readout_metrics(metrics=metrics, stage=stage, *args, **kwargs)
        if 'level_ce' in metrics.keys():
            self.log(f'CE/{stage}', metrics['level_ce'], batch_size=self.data_args['batch_size'])
        return metrics

    def conv_step(self, *args, **kwargs):
        x = kwargs['x']
        conv = kwargs['conv']
        edge_index_dict = kwargs['edge_index_dict']
        x['node'] = x['node'] + conv(x, edge_index_dict)
        return x

    def get_node_embeddings(self, batch):
        emb = super().get_node_embeddings(batch=batch)
        # todo 设置固定、动态的时间段特征以及相关的混合学习机制
        normal = torch.normal(0, 1, size=emb['time'].size(), device=emb['time'].device)
        emb['time'] = self.time_emb_fuse_mlp(torch.cat([emb['time'], normal], 1))
        for k in emb.keys():
            emb[k] = self.noise(emb[k])
        return AdditiveDict(emb)

    def get_edge_embeddings(self, batch, *args, **kwargs):
        return AdditiveDict()

    def training_step(self, batch, batch_index) -> Optional[STEP_OUTPUT]:
        y = batch.y.float()
        out = self(batch)
        metrics = self.metrics(stage='train', y=y, batch=batch, **out)
        loss = metrics['loss']
        self.log(f'loss/train', loss, batch_size=self.data_args['batch_size'])
        self.log(f'msle/train', metrics['msle'], batch_size=self.data_args['batch_size'],
                 )
        return loss

    def on_after_backward(self) -> None:
        self.log_weights(grad=True)

    def validation_step(self, batch, batch_index) -> Optional[STEP_OUTPUT]:
        y = batch.y
        out = self(batch)
        metrics = self.metrics(stage='val', y=y, batch=batch, **out)
        loss = metrics['loss']
        msle = metrics['msle']
        self.log(f'loss/val', loss, batch_size=self.data_args['batch_size'])
        self.log(f'msle/val', metrics['msle'], batch_size=self.data_args['batch_size'])
        return {'loss': loss,
                'batch': batch,
                'y': y,
                'msle': msle,
                'pred': out['pred']}

    def test_step(self, batch, batch_index) -> Optional[STEP_OUTPUT]:
        y = batch.y
        out = self(batch)
        metrics = self.metrics(stage='test', y=y, batch=batch, **out)
        loss = metrics['loss']
        msle = metrics['msle']
        self.log('loss/test', loss, batch_size=self.data_args['batch_size'])
        self.log('msle/test', msle, batch_size=self.data_args['batch_size'])
        self.log('hp_metric', msle, batch_size=self.data_args['batch_size'])
        return {'loss': loss,
                'batch': batch,
                'y': y,
                'msle': msle,
                'pred': out['pred']}

    def validation_epoch_end(self, outputs) -> None:
        if self.logger:
            data, msle = collect_result(outputs)
            title = None
            title = '%s %s %s MSLE(%.3f)' % (osp.split(self.logger.save_dir)[-1],
                                             self.logger.name, self.logger.version, sum(msle) / len(msle))
            axs = plot_result(data, title=title, show=False)
            # test trained model
            tensorboard = self.logger.experiment
            tensorboard.add_figure('result/val', plt.gcf(), self.trainer.current_epoch)  # log on tensorboard


    def test_epoch_end(self, outputs) -> None:
        data, msle = collect_result(outputs)
        title = None
        if self.logger:
            title = '%s %s %s MSLE(%.3f)' % (osp.split(self.logger.save_dir)[-1],
                                             self.logger.name, self.logger.version, sum(msle)/len(msle))
        axs = plot_result(data, title=title, show=False)
        if self.logger:
            # test trained model
            save_path = osp.join(self.logger.save_dir, self.logger.name, f'version_{self.logger.version}',
                                 'test_result.png')
            plt.savefig(save_path)
            tensorboard = self.logger.experiment
            tensorboard.add_figure('result/test', plt.gcf())    # log on tensorboard
        else:
            # test loaded (from checkpoint) model
            plt.show()
        pass


    def get_meta_edge(self, edge_name: str) -> tuple:
        for s, e, t in self.meta_edges:
            if e == edge_name:
                return (s, e, t)
        return None


class TimeAttendPredictor(HeteroGCNPopularityPredictor):
    name = 'TimeAttend'

    def __init__(self,
                 num_time_nodes=0, num_heads=1,
                 time_loss_weight=None, time_cs_offset=None,
                 bro_loss_weight=None,
                 adjacency_time_loss=True, ends_time_loss=True, spaced_time_loss=True,
                 *args, **kwargs
                 ):
        self.bro_loss_weight = bro_loss_weight if bro_loss_weight else 0
        if time_loss_weight is None or time_loss_weight == 0:
            adjacency_time_loss = ends_time_loss = spaced_time_loss = False
        self.num_time_nodes = num_time_nodes
        self.num_heads = num_heads
        self.time_cs_offset = time_cs_offset
        self.adjacency_time_loss = adjacency_time_loss
        self.ends_time_loss = ends_time_loss
        self.spaced_time_loss = spaced_time_loss
        self.time_loss_weight = time_loss_weight if time_loss_weight else 0
        super().__init__(*args, **kwargs)
        self.time_base_embeddings = nn.Embedding(num_time_nodes, self.in_feats)
        # to do: 用户节点 -> 时间节点：时间段特征基础 + Attention(q=time_embedding, k=time_stamp, v=user_embedding) 的时间段特征聚合
        # to do: 时间节点 -> 时间节点：结合RNN、Attention计算时序衰减的系数
        # to do: 时间节点 -> 用户节点：结合Attention，应用（聚合）时间节点中的时序衰减系数并计算衰减后的特征

        '''
        RNN相关模块
        share_time_module：是否所有GCN层后共享同一个RNN相关模块
        num_time_modules：RNN相关模块数
        '''
        self.gcn_norm = nn.ModuleDict({f'{nt}__{i}': nn.LayerNorm(self.in_feats)
                                       for nt in ('time', 'user')
                                       for i in range(self.num_gcn_layers)})

        self.save_hyperparameters('num_time_nodes', 'num_heads',
                                  'time_loss_weight', 'time_cs_offset',
                                  'bro_loss_weight',
                                  'adjacency_time_loss', 'ends_time_loss', 'spaced_time_loss',
                                  'meta_edges',
                                  ignore='noise')
        # self.reset_parameters()

    def get_convs(self):
        convs = []
        meta_nodes = set()
        edge_with_time = ['contain', 'postat', 'repost', ]
        for me in self.meta_edges:
            meta_nodes.update([me[0], me[2]])
        meta_data = (list(meta_nodes), self.meta_edges)
        for i in range(self.num_gcn_layers):
            in_channels = self.in_feats if i != 0 else self.in_feats
            out_channels = self.in_feats
            conv = HANConv(in_channels=in_channels, out_channels=out_channels, metadata=meta_data,
                           heads=self.num_heads, dropout=self.dropout)
            convs.append(conv)
        return convs

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super(TimeAttendPredictor, cls).add_model_specific_args(parent_parser)
        group = parser.add_argument_group('TimeAttendPredictor')
        group.add_argument('--num_time_nodes', type=int, default=0, help='number of time nodes.')
        group.add_argument('--num_heads', type=int, default=1, help='number of attention heads.')
        # group.add_argument('--num_ca_layers', type=int, default=1, help='number of co-attention layer for readout.')
        group.add_argument('--time_loss_weight', type=float, default=0, help='wights of time nodes similarity.')
        group.add_argument('--bro_loss_weight', type=float, default=0,
                           help='wight of Batch Representation Orthogonality penalty, from the “Improving Molecular '
                                'Graph Neural Network Explainability with Orthonormalization and Induced Sparsity” '
                                'paper.')
        return parser

    def conv_step(self, *args, **kwargs):
        x = kwargs['x']
        conv = kwargs['conv']
        edge_index_dict = kwargs['edge_index_dict']
        conv_step = kwargs.get('conv_step', 0)
        size_dict = {
            (s, e, t): (x['node'][s].size(0), x['node'][t].size(0))
            for (s, e, t) in self.meta_edges
        }
        conv_result = conv(x_dict=x['node'], edge_index_dict=edge_index_dict,
                           # edge_attr_dict=x['edge'], size_dict=size_dict,
                           )
        for k, v in conv_result.items():
            if k in x['node'].keys():
                x['node'][k] = x['node'][k] + v
        kwargs['x'] = x
        for k, v in x['node'].items():
            # x['node'][k] = self.gcn_norm[f'{k}__{conv_step}'](v)
            x['node'][k] = self.activate_func(v)
            x['node'][k] = self.dropout_func(v)
        return x

    # def agg_subcas_time(self, x, batch, reduce='sum'):
    #     '''
    #     对于每一个多源级联，纵向地聚合每一个子级联中相同位置的时间节点。
    #     由于不同级联中，子级联数量不统一，横向的聚合在操作上有难度。
    #     :param x: embedding dict
    #     :param batch: batch data
    #     :param reduce: reduce function, any can be passed to scatter function.
    #     :return: aggregated embedding of time nodes for each sub-cascade .
    #     '''
    #     x_time = x['node']['time']
    #     batch_index = batch.batch_dict['time']
    #     node_id = batch['time']['time_node_id']
    #     batch_time_idx = batch_index * self.num_time_nodes + node_id
    #     scattered_time = scatter(x_time, batch_time_idx, 0, reduce=reduce)
    #     # The shape of scattered_time is [batch_size, num_time_nodes, in_feats]
    #     scattered_time = scattered_time.view([-1, self.num_time_nodes, self.in_feats])
    #     return scattered_time

    def metrics(self, stage, *args, **kwargs):
        metrics = super(TimeAttendPredictor, self).metrics(stage, *args, **kwargs)
        loss = metrics['loss']
        x_node = kwargs['x']['node']
        batch = kwargs['batch']
        batch_dict = batch.batch_dict
        if self.bro_loss_weight > 0:
            bro_loss = sum([bro(x_node[k], batch_dict[k]) for k in x_node.keys()])
            bro_loss *= self.bro_loss_weight
            self.log(f'bro_loss/{stage}', bro_loss)
            metrics['loss'] = metrics['loss'] + bro_loss
            metrics['bro'] = bro_loss
        if self.time_loss_weight:
            x_time = x_node['time'].view(-1, self.num_time_nodes, self.in_feats)
            offsets = [self.time_cs_offset] if self.time_cs_offset else []
            time_loss = torch.zeros([x_time.shape[0], 1], device=x_time.device)
            if self.adjacency_time_loss:
                # 最大化相邻时间节点的余弦相似度
                # todo: 考虑使用交叉熵等误差
                adjacent_time_cs = 1 - F.cosine_similarity(x_time[:, :-1], x_time[:, 1:], 2)
                time_loss += torch.sum(adjacent_time_cs)
            if self.spaced_time_loss:
                # 最小化间隔时间节点之间余弦相似度
                for ofs in offsets:
                    spaced_time_cs = 1 + F.cosine_similarity(x_time[:, :-ofs], x_time[:, ofs:], 2)
                    time_loss += torch.sum(spaced_time_cs)
            if self.ends_time_loss:
                # 最小化首尾时间节点之间余弦相似度
                ending_time_cs = 1 + F.cosine_similarity(x_time[:, 0], x_time[:, -1], 1)
                time_loss += torch.sum(ending_time_cs)
            cs = self.time_loss_weight * time_loss.mean()
            if stage == 'train':
                self.log(f'time_cosine_similarity/{stage}', cs, on_step=True, prog_bar=False)
            metrics['loss'] = metrics['loss'] + cs
            metrics['cs'] = cs
        return metrics


class TimestampAttendPredictor(TimeAttendPredictor):
    name = 'TimestampAttend'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        norm_dict = {}
        node_types = set()
        self.ts_lin = nn.ModuleDict()
        self.ts_norm = LayerNorm(1)
        for u, e, v in self.meta_edges:
            node_types.update([u, v])
            self.ts_lin['__'.join([u, e, v])] = nn.Linear(1, self.in_feats)
        for i in range(self.num_gcn_layers):
            for me in self.meta_edges:
                k = '__'.join(me) + f'__{i}'
                # layer_norm_dict[k] = nn.LayerNorm(self.in_feats)
                norm_dict[k] = GraphNorm(self.in_feats)
            for nt in node_types:
                k = f'{nt}__{i}'
                # layer_norm_dict[k] = nn.LayerNorm(self.in_feats)
                norm_dict[k] = GraphNorm(self.in_feats)
        self.gcn_norm = nn.ModuleDict(norm_dict)

    def get_convs(self):
        convs = []
        meta_nodes = set()
        edge_with_time = [('time', 'contain', 'user'), ('user', 'postat', 'time'), ('time', 'repost', 'user'), ]
        for me in self.meta_edges:
            meta_nodes.update([me[0], me[2]])
        meta_data = (list(meta_nodes), self.meta_edges)
        for i in range(self.num_gcn_layers):
            in_channels = self.in_feats if i != 0 else self.in_feats
            out_channels = self.in_feats
            conv = HTSAConv(in_feats=in_channels, out_feats=out_channels,
                            num_heads=self.num_heads,
                            metadata=meta_data,
                            edges_with_ts=[],
                            # edges_with_ts=edge_with_time,
                            dropout=self.dropout,
                            )

            convs.append(conv)
        return convs

    def conv_step(self, *args, **kwargs):
        x = kwargs['x']
        conv = kwargs['conv']
        edge_index_dict = kwargs['edge_index_dict']
        conv_step = kwargs.get('conv_step', 0)
        batch = kwargs['batch']
        timestamp_dict = {}

        for (s, e, t) in self.meta_edges:
            if 'time' in batch[e].keys():
                timestamp = torch.unsqueeze(batch[e]['time'].float(), 1)
                batch_dict = batch.batch_dict[s].index_select(0, batch[e]['edge_index'][0])
                timestamp_dict[(s, e, t)] = self.ts_norm(timestamp, batch_dict)

        conv_result, edge_index_dict = conv(x_dict=x['node'], edge_index_dict=edge_index_dict,
                                            edge_x_dict=x['edge'],
                                            timestamp_dict=timestamp_dict,
                                            )
        for t, x_dict in x.items():
            for k, v in x_dict.items():
                # residual connection
                if self.dropout_edge == 0:
                    for i in range(len(conv_result[t][k])):
                        if len(v[0]) == len(conv_result[t][k][i]):
                            v = v + conv_result[t][k][i]
                else:
                    v = None
                    for i in range(len(conv_result[t][k])):
                        v = v + conv_result[t][k][i]
                norm_key = (k if isinstance(k, str) else '__'.join(k)) + f'__{conv_step}'
                if t == 'node':
                    batch_dict = batch.batch_dict[k]
                else:
                    batch_dict = batch.batch_dict[k[0]][edge_index_dict[k][0]]
                v = self.gcn_norm[norm_key](v, batch_dict)
                v = self.activate_func(v)

                x[t][k] = self.dropout_func(v)
        kwargs['x'] = x
        kwargs['edge_index_dict'] = edge_index_dict
        self.time_embed(*args, **kwargs)
        return x, edge_index_dict

    def get_edge_embeddings(self, batch, *args, **kwargs):
        # todo 实现边的特征存储以及获取。
        if 'edge_index_dict' in kwargs.keys():
            edge_index_dict = kwargs['edge_index_dict']
        else:
            edge_index_dict = batch.edge_index_dict
        edge_embedding_dict = {}
        for (u, e, v), edge_index in edge_index_dict.items():
            num_edges = edge_index.size(-1)
            edge_embedding_dict[(u, e, v)] = torch.normal(0, 1, [num_edges, self.in_feats],
                                                          device=self.device)
            if 'time' in batch[e].keys():
                # 直接使用 edge_index 可能会包含自动添加的自环边
                # 使用 batch[e]['edge_index'] 过滤自环
                batch_dict = batch.batch_dict[u].index_select(0, batch[e]['edge_index'][0])
                timestamp = batch[e]['time'].float()
                timestamp = torch.unsqueeze(timestamp, 1)
                timestamp = self.ts_norm(timestamp, batch_dict)
                lin = self.ts_lin['__'.join([u, e, v])]
                x_timestamp = lin(timestamp)
                # 对于有时间戳的边，增加时间戳相关的特征
                upper = min(edge_embedding_dict[(u, e, v)].size(0), timestamp.size(0))
                edge_embedding_dict[(u, e, v)][:upper] = edge_embedding_dict[(u, e, v)][:upper] + x_timestamp[:upper]
        return edge_embedding_dict
