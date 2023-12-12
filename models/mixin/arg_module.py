f'''
可通过参数初始化的模块
通过参数来初始化相关模块，
'''
__all__ = ['ModuleEmbeddings', 'ModuleReadout', 'ModuleTimeEmbed']

from argparse import ArgumentParser
from copy import deepcopy

from torch import nn

from nn import LearnableEmbeddings, CollatedEmbeddings


class ModuleEmbeddings(nn.Module):
    def __init__(self, total_nodes, total_edges=None, learnable_embedding=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_embeddings = None
        self.total_nodes = total_nodes
        self.total_edges = total_edges
        if learnable_embedding:
            self.init_embeddings = LearnableEmbeddings(total_nodes=self.total_nodes, total_edges=self.total_edges,
                                                       *args, **kwargs)
        else:
            self.init_embeddings = CollatedEmbeddings(*args, **kwargs)
        self.save_hyperparameters('learnable_embedding')

    @classmethod
    def add_model_specific_args(self, parser: ArgumentParser):
        parser = super().add_model_specific_args(parser)
        group = parser.add_argument_group('ModuleEmbeddings')
        group.add_argument('--learnable_embedding', default=False, action='store_true',
                           help='use learnable embedding, random embedding for each batch if not.')
        return parser

    def get_node_embeddings(self, *args, **kwargs):
        return self.init_embeddings.get_node_embeddings(*args, **kwargs)

    def get_edge_embeddings(self, *args, **kwargs):
        return self.init_embeddings.get_edge_embeddings(*args, **kwargs)

    def get_embeddings(self, *args, **kwargs) -> dict:
        emb = {
            'node': self.get_node_embeddings(*args, **kwargs),
            'edge': self.get_edge_embeddings(*args, **kwargs),
        }
        return emb


class ModuleTimeEmbed(nn.Module):
    def __init__(self, time_module=None,
                 share_time_module=False, num_time_module_layers=2,
                 time_decay_pos=None, time_decay_cum=False,
                 *args, **kwargs):
        self.time_module = time_module
        self.share_time_module = share_time_module
        self.num_time_module_layers = num_time_module_layers
        self.time_decay_pos = time_decay_pos
        self.time_decay_cum = time_decay_cum
        self.time_decay = None
        super(ModuleTimeEmbed, self).__init__(*args, **kwargs)
        if self.time_decay_pos != 'None':
            from nn import TimeDecay
            self.time_decay = TimeDecay(in_feats=self.in_feats, num_time_nodes=self.num_time_nodes,
                                        cumulate=self.time_decay_cum)
        else:
            self.time_decay_pos = None
        num_time_modules = 1 if self.share_time_module else self.num_gcn_layers
        time_module_obj = None
        all_args = kwargs.get('all_args', self.all_args)
        if self.time_module == 'rnn':
            from nn import TimeRNN
            time_module_obj = TimeRNN(num_rnn_layers=self.num_time_module_layers, **all_args)
        elif self.time_module == 'transformer':
            from nn import TimeTransformer
            time_module_obj = TimeTransformer(num_encoder_layers=self.num_time_module_layers,
                                              **all_args)
        else:
            self.time_modules = None
        if time_module_obj is not None:
            self.time_modules = nn.ModuleList([deepcopy(time_module_obj) for i in range(num_time_modules)])
        self.save_hyperparameters('time_module', 'share_time_module', 'num_time_module_layers', 'time_decay_pos',
                                  'time_decay_cum')

    @classmethod
    def add_model_specific_args(self, parser: ArgumentParser):
        parser = super().add_model_specific_args(parser)
        group = parser.add_argument_group('ModuleTimeEmbed')
        group.add_argument('--time_module', default=None,
                           choices=['None', 'rnn', 'transformer'],
                           help='time embed module .')
        group.add_argument('--num_time_module_layers', default=2, type=int,
                           help='the number of layers of time module, such as rnn layers or transformer encoder layers')
        group.add_argument('--share_time_module', default=False, action='store_true',
                           help='use learnable embedding, random embedding for each batch if not.')
        group.add_argument('--time_decay_pos', default=None, choices=['None', 'head', 'tail', 'all'],
                           help='position of time decay, ')
        group.add_argument('--time_decay_cum', default=False, action='store_true',
                           help='accumulate decay weights for each time interval if ture,'
                                ' else time decay independently.')
        return parser

    def time_embed(self, *args, **kwargs):
        x = kwargs['x']
        all_x = isinstance(x, dict)
        if all_x:
            time_x = x['node']['time']
        else:
            time_x = x
        conv_step = kwargs.get('conv_step', 0)
        total_steps = self.num_gcn_layers
        time_module_index = 0 if self.share_time_module else conv_step
        if self.time_modules:
            time_module = self.time_modules[time_module_index]
            time_x = time_module(time_x)

        if self.time_decay is not None and (self.time_decay_pos == 'all' or
                                            (self.time_decay_pos == 'head' and conv_step == 0) or
                                            (self.time_decay_pos == 'tail' and conv_step == total_steps)):
            time_x = self.time_decay(time_x)

        if all_x:
            x['node']['time'] = time_x
        return time_x


class ModuleReadout(nn.Module):
    def __init__(self, readout='ca',
                 num_readout_layers=3,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.readout = None
        self.num_readout_layers = num_readout_layers
        all_args = kwargs.get('all_args', self.all_args)
        if readout == 'ca':
            from nn import CoAttentionReadout
            self.readout = CoAttentionReadout(**all_args)
        elif readout == 'mlp':
            from nn import MLPReadout
            self.readout = MLPReadout(**all_args)
        elif readout == 'uta':
            from nn import UserTimeAttentionReadout
            self.readout = UserTimeAttentionReadout(**all_args)
        elif readout == 'ml':
            from nn import MultiLevelReadout
            self.readout = MultiLevelReadout(**all_args)
        else:
            from nn import CoAttentionReadout
            self.readout = CoAttentionReadout(num_readout_layers=num_readout_layers, *args, **kwargs)
        # self.readout.predictor = self
        self.save_hyperparameters('readout', 'num_readout_layers')

    @classmethod
    def add_model_specific_args(self, parser: ArgumentParser):
        parser = super().add_model_specific_args(parser)
        group = parser.add_argument_group('ModuleReadout')
        group.add_argument('--readout', type=str, default='ca', choices=['ca', 'mlp', 'uta', 'us', 'ml'],
                           help='the readout module.')
        group.add_argument('--num_readout_layers', type=int, default=1,
                           help='the number of internal layers of readout module.')
        group.add_argument('--num_levels', type=int, default=10,
                           help='the number of levels in multi-level readout.')
        return parser
