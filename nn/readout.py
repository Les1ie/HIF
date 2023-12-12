__all__ = ['BaseReadout', 'CoAttentionReadout', 'MLPReadout', 'UserTimeAttentionReadout', 'MultiLevelReadout']

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import unbatch, unbatch_edge_index
from torch_scatter import scatter


class BaseReadout(nn.Module):
    def __init__(self, in_feats, out_feats,
                 num_time_nodes, num_heads, num_readout_layers,
                 dropout, activate_cls,
                 mode='batch',
                 *args, **kwargs):
        self.in_feats = in_feats
        self.hid_feats = 3 * in_feats
        self.out_feats = out_feats
        self.num_time_nodes = num_time_nodes
        self.num_heads = num_heads
        self.mode = mode
        self.batch_mode = self.mode == 'batch'
        self.num_readout_layers = num_readout_layers
        assert num_readout_layers > 0, \
            'the number of readout mlp output_mlp_layers must be greater than zero, got %d' % num_readout_layers
        self.dropout = dropout
        self.activate_cls = activate_cls
        super(BaseReadout, self).__init__()

    # 用于 readout 的通用方法
    def get_user(self, x, *args, **kwargs):
        if self.batch_mode:
            batch = kwargs.get('batch')
            user_batch_dict = batch.batch_dict['user']
            user_x = x['node']['user']
            return user_x, user_batch_dict
        else:
            return x['node']['user']

    def get_sender_receiver(self, x, *args, **kwargs):
        if self.batch_mode:
            batch = kwargs.get('batch')
            user_batch_dict = batch.batch_dict['user']
            repost_edges = batch['user', 'repost', 'user']['edge_index']
            sender, receiver = repost_edges
            sender_x = x['node']['user'].index_select(0, sender)
            sender_batch_dict = user_batch_dict.index_select(0, sender)
            receiver_x = x['node']['user'].index_select(0, receiver)
            receiver_batch_dict = user_batch_dict.index_select(0, receiver)
            return sender_x, sender_batch_dict, receiver_x, receiver_batch_dict
        else:
            edge_index = kwargs.get('edge_index')
            sender_x = []
            receiver_x = []
            for i in range(len(x['node']['user'])):
                e = edge_index['user', 'repost', 'user'][i]
                sender_x.append(x['node']['user'][i].index_select(0, e[0]))
                receiver_x.append(x['node']['user'][i].index_select(0, e[1]))

            return sender_x, receiver_x

    def get_time(self, x, *args, **kwargs):
        # if self.batch_mode:
        #     batch = kwargs.get('batch')
        #     time_factor = scatter(x['node']['time'], batch.batch_dict['time'], dim=0)
        #     time_batch_dict = batch.batch_dict['time']
        #     return scatter(x['node']['time']
        # else:
        return x['node']['time']

    def get_follower(self, x, *args, **kwargs):
        if self.batch_mode:
            batch = kwargs.get('batch')
            follow_edges = batch['user', 'follow', 'user']['edge_index']
            user_batch_dict = batch.batch_dict['user']
            follower = follow_edges[1]
            follower_x = x['node']['user'].index_select(0, follower)
            follower_batch_dict = user_batch_dict.index_select(0, follower)
            return follower_x, follower_batch_dict
        else:
            edge_index = kwargs.get('edge_index')
            follower_x = []
            for i in range(len(x['node']['user'])):
                e = edge_index['user', 'follow', 'user'][i]
                follower_x.append(x['node']['user'][i].index_select(0, e[1]))
            return follower_x

    def readout_metrics(self, metrics, *args, **kwargs):
        return metrics

    def unbatch_graph(self, *args, **kwargs):
        '''
        将 kwargs 中的图进行 unbatch
        :param kwargs:
        :return: unbatch 后的点、边特征和 edge_index
        '''
        batch = kwargs['batch']
        batched_x = kwargs.get('x')
        batched_edge_index = {et: batch[et]['edge_index'] for et in batch.edge_types}
        x = {'node': {k: unbatch(v, batch.batch_dict[k]) for k, v in batched_x['node'].items()},
             # 'edge': {k: unbatch(v, batch.batch_dict[k[0]].index_select(0, batched_edge_index[k][0]))
             #          for k, v in batched_x['edge'].items()},
             }
        edge_index = {k: unbatch_edge_index(v, batch.batch_dict[k[0]]) for k, v in batched_edge_index.items()}
        return edge_index, x


class MLPReadout(BaseReadout):
    name = 'mlp'

    def __init__(self, *args, **kwargs):
        super(MLPReadout, self).__init__(*args, **kwargs)

        self.node_roles = ['sender', 'receiver', 'follower',
                           # 'time'
                           ]
        mlp_dict = {}
        for role in self.node_roles:
            layers = self.get_mlp_layers(d_in=self.in_feats, d_hid=self.in_feats, d_out=self.in_feats, last_layer=False)
            # layers.append(nn.ReLU())
            mlp_dict[role] = nn.Sequential(*layers)
        self.role_mlp_dict = nn.ModuleDict(mlp_dict)

        # out_mlp_in_feats = 1 + self.in_feats * 3
        output_mlp_layers = self.get_mlp_layers(d_in=4 * self.in_feats, d_hid=self.hid_feats, d_out=self.out_feats,
                                                last_layer=True)
        # output_mlp_layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*output_mlp_layers)
        self.layer_norm = nn.LayerNorm(1)

    def get_mlp_layers(self, d_in=1, d_hid=None, d_out=None,
                       last_layer=True):
        if not d_hid:
            d_hid = self.hid_feats
        if not d_out:
            d_out = self.out_feats
        output_mlp_in_feats = d_in
        output_mlp_layers = []
        for i in range(self.num_readout_layers):
            dim_in, dim_out = d_hid, d_hid
            if i == 0:
                dim_in = output_mlp_in_feats
            if i == self.num_readout_layers - 1:
                dim_out = d_out

            output_mlp_layers.append(nn.Linear(dim_in, dim_out))

            if i + 1 < self.num_readout_layers:
                output_mlp_layers.extend([self.activate_cls(),
                                          nn.LayerNorm(d_hid),
                                          nn.Dropout(self.dropout)])
            elif last_layer:
                output_mlp_layers.extend([
                    # nn.CELU(0.5),
                    nn.Flatten(0)])
        return output_mlp_layers

    def forward(self, *args, **kwargs):
        batched_x = kwargs['x']
        if self.batch_mode:
            x = kwargs['x']
            batch = kwargs['batch']
            sender_x, sender_batch_dict, receiver_x, receiver_batch_dict = self.get_sender_receiver(x, batch=batch)
            follower_x, follower_batch_dict = self.get_follower(x, batch=batch)

            sender_x = sender_x + self.role_mlp_dict['sender'](receiver_x)
            receiver_x = receiver_x + self.role_mlp_dict['receiver'](receiver_x)
            follower_x = follower_x + self.role_mlp_dict['follower'](follower_x)
            sender_factor = scatter(sender_x, sender_batch_dict, dim=0)
            receiver_factor = scatter(receiver_x, receiver_batch_dict, dim=0)
            follower_factor = scatter(follower_x, follower_batch_dict, dim=0)
        else:
            if isinstance(kwargs['x']['node']['user'], torch.Tensor):
                edge_index, x = self.unbatch_graph(**kwargs)
            else:
                edge_index = kwargs.get('edge_index')
                x = kwargs.get('x')
            sender_x, receiver_x, = self.get_sender_receiver(x, edge_index=edge_index)
            follower_x = self.get_follower(x, edge_index=edge_index)
            sender_factor = torch.stack([torch.sum(si, dim=0) for si in sender_x], dim=0)
            receiver_factor = torch.stack([torch.sum(ri, dim=0) for ri in receiver_x], dim=0)
            follower_factor = torch.stack([torch.sum(fi, dim=0) for fi in follower_x], dim=0)

        # user_factor = torch.cat([sender_factor, receiver_factor], dim=1)
        # user_factor = scatter(x['node']['user'], batch.batch_dict['user'], dim=0)
        # summed_factor = torch.sum(user_factor, dim=1, keepdim=True)
        # summed_factor = summed_factor + summed_factor * torch.sum(follower_factor, dim=1, keepdim=True)
        # time_factor = scatter(x['node']['time'], batch.batch_dict['time'], dim=0)
        time_factor = self.get_time(**kwargs).view(-1, self.num_time_nodes, self.in_feats)[:, -1, :]
        # time_factor = time_factor + self.role_mlp_dict['time'](time_factor)
        cat_x = torch.cat([sender_factor, receiver_factor, follower_factor, time_factor], dim=1)
        # cat_x = torch.cat([summed_factor, user_factor, time_factor], dim=1)
        # cat_x = self.layer_norm(summed_factor + summed_factor * torch.sum(time_factor, dim=1, keepdim=True))
        pred = self.mlp(cat_x)
        # pred = self.mlp(cat_x) + torch.squeeze(summed_factor, dim=1)
        # pred = F.celu(self.mlp(cat_x) + torch.squeeze(summed_factor, 1), 0.5)
        return {'pred': pred,
                'x': batched_x}


class UserTimeAttentionReadout(BaseReadout):
    name = 'uta'

    def __init__(self, *args, **kwargs):
        super(UserTimeAttentionReadout, self).__init__(*args, **kwargs)
        self.user_time_attention = nn.MultiheadAttention(embed_dim=self.in_feats, num_heads=self.num_heads,
                                                         batch_first=True)
        self.scale_factor = torch.tensor(self.in_feats)
        mlps = []
        for i in range(self.num_readout_layers):
            first_layer = i == 0
            last_layer = i == self.num_readout_layers - 1
            d_in = self.in_feats if first_layer else self.hid_feats
            d_out = self.out_feats if last_layer else self.hid_feats
            mlps.append(nn.Linear(d_in, d_out))
            if not last_layer:
                mlps.append(nn.LazyBatchNorm1d())
                mlps.append(nn.Dropout(self.dropout))
                mlps.append(self.activate_cls())
            else:
                mlps.append(nn.Flatten(0))
        self.user_mlp = nn.Sequential(
            nn.Linear(self.in_feats, self.in_feats),
            nn.LayerNorm(self.in_feats), nn.Dropout(self.dropout), self.activate_cls(),
            nn.Linear(self.in_feats, self.in_feats)
        )
        self.readout_mlp = nn.Sequential(*mlps)
        self.layer_norm = nn.LayerNorm(self.in_feats)

    def forward(self, *args, **kwargs):
        x, batch = kwargs['x'], kwargs['batch']
        user_x = x['node']['user']
        time_x = x['node']['time']
        user_batch_dict = batch.batch_dict['user']
        time_batch_dict = batch.batch_dict['time']
        user_factor = scatter(user_x, user_batch_dict, dim=0)

        summed_user = torch.sum(self.user_mlp(user_factor), dim=1, keepdim=False) / self.scale_factor
        unsqueezed_user_factor = torch.unsqueeze(user_factor, 1)
        unsqueezed_time_x = time_x.view([-1, self.num_time_nodes, self.in_feats])
        attended_user, _ = self.user_time_attention(unsqueezed_user_factor, unsqueezed_time_x, unsqueezed_time_x,
                                                    need_weights=False)
        # [batch_size, 1, in_feats]，对第2维求和并且不 keep_dim=False 返回 [batch_size, 1]
        readout_factor = self.layer_norm(user_factor + torch.squeeze(attended_user, dim=1))
        pred = summed_user + self.readout_mlp(readout_factor)
        return {'pred': pred,
                'x': x}


class CoAttentionReadout(BaseReadout):
    name = 'ca'

    def __init__(self, *args, **kwargs):
        super(CoAttentionReadout, self).__init__(*args, **kwargs)
        from nn import CoAttentionEncoderDecoder
        self.co_attention = CoAttentionEncoderDecoder(feats=self.in_feats,
                                                      num_layers=self.num_readout_layers,
                                                      num_heads=self.num_heads, dropout=self.dropout)
        num_factors = 4
        factor_feats = self.in_feats * num_factors
        self.out_mlp = nn.Sequential(
            nn.Linear(factor_feats, factor_feats),
            nn.LayerNorm(factor_feats),
            nn.Dropout(self.dropout), self.activate_cls(),
            nn.Linear(factor_feats, self.out_feats),
            nn.CELU(0.5),
            nn.Flatten(0)
        )

    def batch_padding(self, x, batch):
        '''
        将所有图中用户的数量填充至batch中的最大值
        @param x:
        @param batch:
        @return:
        '''
        x_time = x['node']['time'].view([-1, self.num_time_nodes, self.in_feats])
        user_batch_index = batch.batch_dict['user']
        batch_size = batch.num_graphs
        x_user = x['node']['user']
        ones = torch.ones_like(batch.batch_dict['user'])
        num_users = scatter(ones, batch.batch_dict['user'], 0)
        max_users = max(num_users)
        mask_factory_kwarg = {'dtype': bool, 'device': x_user.device}
        # batch_size * num_heads, max_users, num_time_nodes
        attn_mask_tu = torch.zeros([batch_size * self.num_heads, self.num_time_nodes, max_users], **mask_factory_kwarg)
        attn_mask_user = torch.zeros([batch_size * self.num_heads, max_users, max_users], **mask_factory_kwarg)
        key_padding_mask_user = torch.zeros([batch_size, max_users], **mask_factory_kwarg)
        padded = torch.zeros([batch_size, max_users, self.in_feats], device=x_user.device)
        tot = 0
        for i in range(batch_size):
            n = num_users[i]
            padded[i, 0:n, :] = x_user[tot:tot + n, :]
            attn_mask_tu[i * self.num_heads:(1 + i) * self.num_heads, :, n:] = True
            # 整行为True后续Attention计算softmax会出现nan
            # attn_mask_user[i*self.num_heads:(1+i)*self.num_heads, n:, :] = True
            attn_mask_user[i * self.num_heads:(1 + i) * self.num_heads, :, n:] = True
            key_padding_mask_user[i, n:] = True
            tot += n
        mask = {'attn_mask_tu': attn_mask_tu,
                'attn_mask_user': attn_mask_user,
                'key_padding_mask_user': key_padding_mask_user,
                }
        return padded.contiguous(), x_time.contiguous(), mask

    def forward(self, *args, **kwargs):
        x, batch = kwargs['x'], kwargs['batch']
        x_user, x_time, mask = self.batch_padding(x, batch)

        attn_time, attn_user = self.co_attention(
            x_time, x_user,
            attn_mask_y=mask['attn_mask_user'], attn_mask_xy=mask['attn_mask_tu'],
            key_padding_mask_y=mask['key_padding_mask_user'],
            key_padding_mask_xy=mask['key_padding_mask_user'],
        )
        attn_user = attn_user + x_user
        attn_time = attn_time + x_time
        time_factor = attn_time[:, -1, :]
        user_factor = attn_user.sum(dim=1)
        summed_user_factor = x_user.sum(dim=1)
        cross_factor = user_factor * time_factor
        pred_factor = torch.cat([time_factor, user_factor, cross_factor, summed_user_factor], 1)
        pred = self.out_mlp(pred_factor)
        return {'pred': pred,
                'x': x,
                # 'batched_time': batched_time,
                }


# todo: multi-level readout
# todo：增加额外的档位 loss 计算
class MultiLevelReadout(BaseReadout):
    '''
    多档位预测，在预测前按照一定规则对流行度大小进行分档，对不同档位的特征使用不同的参数/模块进行读出预测。
    '''
    name = 'ml'

    def __init__(self, num_levels=10, *args, **kwargs):
        self.num_levels = num_levels
        super(MultiLevelReadout, self).__init__(*args, **kwargs)
        fac = {'in_feats': self.in_feats, 'out_feats': self.out_feats,
               'num_time_nodes': self.num_time_nodes, 'num_heads': self.num_heads,
               'dropout': self.dropout, 'activate_cls': self.activate_cls}
        fac.update(kwargs)
        # level 分类器
        self.level_base = 2
        self.log_fn_dict = {
            10: torch.log10,
            2: torch.log2,
            torch.e: torch.log
        }
        self.log_fn = self.log_fn_dict[self.level_base]
        hid_feats = 1 + self.in_feats
        self.pow_weight = False
        self.weighted_sum = True
        self.level_mlp = nn.Sequential(
            nn.Linear(hid_feats, hid_feats),
            self.activate_cls(), nn.LayerNorm([hid_feats]), nn.Dropout(self.dropout),
            nn.Linear(hid_feats, self.num_levels),
        )

        # level 预测器
        if self.weighted_sum:
            self.level_readout = nn.Sequential(
                nn.Linear(self.in_feats, self.in_feats),
                self.activate_cls(), nn.LayerNorm([self.in_feats]), nn.Dropout(self.dropout),
                nn.Linear(self.in_feats, self.num_levels),
                nn.ReLU(),
                # nn.Flatten(0)
            )
        else:
            self.level_readout = nn.ModuleList([MLPReadout(mode='unbatch', **kwargs) for i in range(self.num_levels)])
        self.level_ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def get_levels(self, x, batch):
        user_x = x['node']['user']
        unbatched_time_x = self.get_time(x=x, batch=batch).view([-1, self.num_time_nodes, self.in_feats])
        summed_user_x = scatter(user_x, batch.batch_dict['user'], dim=0)
        ones = torch.ones([user_x.size(0), 1], device=user_x.device)
        num_users = scatter(ones, batch.batch_dict['user'], dim=0)
        # x = torch.cat([num_users, summed_user_x], dim=1)
        x = torch.cat([num_users, unbatched_time_x[:, -1, :]], dim=1)
        level_logist = self.level_mlp(x)
        max_logist, levels = torch.max(level_logist, dim=1, )
        return level_logist, levels

    def readout_metrics(self, metrics, *args, **kwargs):
        y = kwargs.get('y')
        max_level = torch.ones_like(y) * (self.num_levels - 1)
        pred_levels = kwargs.get('levels')
        level_logist = kwargs.get('level_logist')
        true_levels = torch.minimum((self.log_fn(y)), max_level).long()
        level_ce = self.level_ce(level_logist, true_levels)
        stage = kwargs.get('stage')
        metrics['level_ce'] = level_ce
        metrics['loss'] = metrics['loss'] + level_ce
        return metrics

    def forward(self, *args, **kwargs):
        batched_x, batch = kwargs['x'], kwargs['batch']
        level_logist, levels = self.get_levels(batched_x, batch)
        edge_index, x = self.unbatch_graph(**kwargs)
        batch_size = len(x['node']['user'])
        if not self.weighted_sum:
            pred = torch.zeros(batch_size, requires_grad=False, device=batched_x['node']['user'].device)
            for i in range(batch_size):
                temp_x = {'node': {'user': [x['node']['user'][i]],
                                   'time': x['node']['time'][i]}}
                temp_edge = {k: [v[i]] for k, v in edge_index.items()}
                for j in range(levels[i]):
                    ro = self.level_readout[j]
                    rst = ro(x=temp_x, edge_index=temp_edge)
                    p = rst['pred']
                    if self.pow_weight:
                        p = self.level_base ** j * p

                    pred[i] = pred[i] + F.leaky_relu(p)
        else:
            bias = scatter(self.level_readout(batched_x['node']['user']), batch['user']['batch'], dim=0)
            if self.pow_weight:
                w = torch.pow(torch.ones_like(bias) * self.level_base,
                              (torch.arange(self.num_levels, device=bias.device)))
                bias = bias * w
            pred = torch.sum(torch.softmax(level_logist, dim=1) * bias, dim=1)
        return {
            'pred': pred,
            'x': batched_x,
            'level_logist': level_logist,
            'levels': levels,
        }
