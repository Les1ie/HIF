import os
import pickle
import time
from collections import defaultdict
from datetime import timedelta
from os import path as osp
from typing import Any, Optional, Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from utils import Indexer
from utils import graph_op as op
from utils.data import pipe

'''
Filters
'''


class LengthFilter(pipe.Filter):
    '''
    Filter Dataframes by length.
    '''

    def __init__(self, min_length=2, max_length=None, **kwargs):
        super(LengthFilter, self).__init__(**kwargs)
        self.max_length = max_length
        self.min_length = min_length
        self.context_state['cascade_length'] = []
        none_zero_msg = "Expected {} length is None or an integer greater than zero."
        if min_length:
            assert min_length > 0, none_zero_msg.format('minimum')
        if max_length:
            assert max_length > 0, none_zero_msg.format('maximum')
        if min_length and max_length:
            assert max_length >= min_length, \
                "Expected maximum length is not less than minimum length."

    @property
    def filename_tag(self):
        tag = super().filename_tag
        if self.min_length > 2:
            tag += f'GT{self.min_length}'  # greater than
        if self.max_length:
            tag += f'LT{self.max_length}'  # less than
        return tag

    def filter(self, item, state=None, dataset=None, *args, **kwargs) -> bool:
        length = len(item)
        if state:
            state['cascade_length'].append(length)
        if self.max_length and length > self.max_length:
            return True
        if self.min_length and length < self.min_length:
            return True
        return False


class FollowershipFilter(pipe.Filter):
    '''
    按照异构图中关注关系的数量进行过滤，推荐 max_followerships = 20000
    '''

    def __init__(self, min_followerships=10, max_followerships=None, **kwargs):
        super(FollowershipFilter, self).__init__(**kwargs)
        self.min_followerships = min_followerships
        if isinstance(max_followerships, float):
            self.max_followerships = int(max_followerships)
        elif isinstance(max_followerships, str):
            self.max_followerships = int(float(max_followerships))
        else:
            self.max_followerships = max_followerships
        assert self.max_followerships is None or self.max_followerships > 0, \
            "Invalid max follower-ships, got {}({})".format(self.max_followerships, type(self.max_followerships))

    @property
    def filename_tag(self):
        tag = super().filename_tag
        if self.max_followerships is not None:
            t = '{:.0e}'.format(self.max_followerships).replace('e+0', 'e')
            tag += f'MF{t}'
        return tag

    def filter(self, item: HeteroData, state, dataset, *args, **kwargs) -> bool:
        if self.max_followerships is None:
            return False
        num_followerships = item.num_edges_dict['user', 'follow', 'user']
        return num_followerships > self.max_followerships or num_followerships < self.min_followerships


'''
Process Pipelines
'''


class CSV2DataframePipeline(pipe.Pipeline):
    def process(self, item=None, state=None, dataset=None, *args, **kwargs) -> Any:
        if dataset is not None:
            raw_dir = dataset.raw_dir
        elif 'raw_dir' in self.kwargs.keys():
            raw_dir = self.kwargs['raw_dir']
        elif 'raw_dir' in kwargs.keys():
            raw_dir = kwargs['raw_dir']
        else:
            raw_dir = 'data'
        df = pd.read_csv(osp.join(raw_dir, item))
        df['origin_uid'] = df['origin_uid'].fillna(df['uid'])
        df.drop_duplicates(['origin_uid', 'uid'], inplace=True)
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['origin_uid'] = pd.to_numeric(df['origin_uid'], downcast='integer').astype(str)
        # df['origin_uid'] = pd.to_numeric(df['origin_uid'].apply(lambda x: int(x, base=16)), downcast='integer').astype(str)
        # df['origin_uid'] = df['origin_uid']
        df['uid'] = df['uid'].astype(str)
        df.sort_values('created_at', ignore_index=True)
        return df


class Dataframe2SequencePipeline(pipe.Pipeline):
    def __init__(self, relative_timestamp=False, **kwargs):
        super(Dataframe2SequencePipeline, self).__init__(**kwargs)
        self.relative_timestamp = relative_timestamp

    def process(self, item: pd.DataFrame = None, state=None, dataset=None, *args, **kwargs) -> Any:
        item.loc[:, 'created_at'] = item.loc[:, 'created_at'].map(lambda x: int(x.timestamp()))
        try:
            start_timestamp = item.loc[0, 'created_at']
        except Exception as e:
            print(item)
            print(e)
            assert False
        if self.relative_timestamp:
            item['created_at'] -= start_timestamp
        seq = item[['origin_uid', 'uid', 'created_at']].to_numpy().tolist()
        return seq


class FollowerSamplePipeline(pipe.Pipeline):
    '''
    关注采样处理管道，输入<点，边，时间戳>三元组序列，输出为转发、关注数据，具体格式取决于 return_nx_graph 参数。
    Sample follower and return two network.DiGraph.
    '''

    def __init__(self, follow_path, method: str = 'bfs',
                 alpha=80,
                 beta=100,
                 sample_batch=15,
                 source_base=3, reposted_base=1.5, leaf_base=2, cas_base=5,
                 hop=1,
                 reverse:bool=False,
                 return_nx_graph=True, **kwargs):
        f'''
        @param follow_path: 关注边文件路径。
        @param method: 采样方法，可选值为 bfs 或 hop，即使用游走采样或者朴素的按跳采样。
        @param alpha: bfs采样参数，具体见{op.weighted_bfs}相关注释。
        @param beta: bfs采样参数，具体见{op.weighted_bfs}相关注释。
        @param sample_batch: bfs采样参数，具体见{op.weighted_bfs}相关注释。
        @param source_base: bfs采样参数，具体见{op.weighted_bfs}相关注释。
        @param reposted_base: bfs采样参数，具体见{op.weighted_bfs}相关注释。
        @param leaf_base: bfs采样参数，具体见{op.weighted_bfs}相关注释。
        @param hop: hop采样参数，采样的跳数，大于1时采样较慢。
        @param reverse: 是否翻转关注网络
        @param return_nx_graph: 返回值的格式，{True}返回{nx.DiGraph}字典，{False}返回边序列字典。
        @param kwargs: 
        '''
        super(FollowerSamplePipeline, self).__init__(**kwargs)
        self.cas_base = cas_base
        self.reverse = reverse
        self.follow_path = follow_path
        self.hop = hop
        self.alpha = alpha
        self.beta = beta
        self.sample_batch = sample_batch
        self.source_base = source_base
        self.reposted_base = reposted_base
        self.leaf_base = leaf_base
        # assert method in ['bfs', 'hop'], 'Excepted sample method is "bfs" or "hop"'
        self.method = method
        self.keep_nx_graph = return_nx_graph
        self._follow_graph: nx.DiGraph = None
        self._ud_follow_graph: nx.DiGraph = None
        self._reversed_follow_graph: nx.DiGraph = None

    @property
    def filename_tag(self):
        tag = super(FollowerSamplePipeline, self).filename_tag
        if self.method in ['dy', 'bfs']:
            if self.alpha:
                tag += f'A{self.alpha}'
            if self.beta:
                tag += f'B{self.beta}'
            base_tags = 's{}r{}l{}c{}'.format(self.source_base, self.reposted_base, self.leaf_base, self.cas_base)
            base_tags = base_tags.replace('.', '_')
            tag += base_tags
            tag += 'Bfs' if self.method == 'bfs' else 'Dy'
            tag += f'{self.sample_batch}'
        else:
            tag += f'Hop{self.hop}'
        if self.reverse:
            tag += 're'
        return tag

    @property
    def follow_graph(self):
        if self._follow_graph is None:
            self._follow_graph = nx.DiGraph()
            s = time.time()
            print('\nFollower network: loading.', end='')
            with open(self.follow_path, 'r', encoding='utf8') as f:
                lines = f.readlines()
                for line in lines:
                    u, v = line.strip().split(',')
                    self._follow_graph.add_edge(u, v)
            e = time.time()
            nx.set_node_attributes(self._follow_graph, {i for i in self._follow_graph.nodes}, 'raw_id')
            print('\rFollow network: loaded ({} s).'.format(int(e - s)))
        return self._follow_graph

    @property
    def ud_follow_graph(self):
        if self._ud_follow_graph is None:
            self._ud_follow_graph = self.follow_graph.to_undirected(as_view=True)
        return self._ud_follow_graph

    @property
    def reversed_follow_graph(self):
        if self._reversed_follow_graph is None:
            self._reversed_follow_graph = self.follow_graph.reverse()
        return self._reversed_follow_graph

    def process(self, item=None, state=None, dataset=None, *args, **kwargs) -> Any:
        repost_g = nx.DiGraph()
        for u, v, t in item:
            repost_g.add_node(v, time=t, raw_id=v)
            if u not in repost_g.nodes():
                repost_g.add_node(u, time=t, raw_id=u)
            repost_g.add_edge(u, v, time=t)
        follow_g = self.sample_followers(repost_g)
        if self.keep_nx_graph:
            return {
                ('user', 'repost', 'user'): repost_g,
                ('user', 'follow', 'user'): follow_g,
            }
        else:
            return {
                ('user', 'repost', 'user'): item,
                ('user', 'follow', 'user'): list(follow_g.edges()),
            }

    def sample_followers(self, repost_g):
        if self.method == 'bfs':
            sampled_nodes = op.weighted_bfs(repost_g, self.follow_graph, self.ud_follow_graph,
                                            source_base=self.source_base, reposted_base=self.reposted_base,
                                            leaf_base=self.leaf_base,
                                            alpha=self.alpha, beta=self.beta, sample_batch=self.sample_batch)
        elif self.method == 'dy':
            sampled_nodes = op.dynamic_weighted_sampling(repost_g, self.follow_graph,
                                                         source_base=self.source_base, reposted_base=self.reposted_base,
                                                         leaf_base=self.leaf_base, cas_base=self.cas_base,
                                                         alpha=self.alpha, beta=self.beta, sample_batch=self.sample_batch
                                                         )
        else:
            sampled_nodes = op.k_hop(repost_g, self.ud_follow_graph, hop=self.hop)
        # if self.reverse:
        #     source_graph = self.reversed_follow_graph
        # else:
        #     source_graph = self.follow_graph
        # follow_g = source_graph.subgraph(sampled_nodes).copy()

        follow_g = self.follow_graph.subgraph(sampled_nodes)
        if self.reverse:
            follow_g = follow_g.reverse(copy=False)
        return follow_g


class NetworkxGraphs2PygHeteroGraphPipeline(pipe.Pipeline):
    f'''
    类型转换，将{nx.DiGraph}对象字典转换为pyg的{HeteroData}对象。
    Combine multiple {nx.DiGraph} to a torch_geometric.HeteroData.
    '''

    def __init__(self, user_indexer, reindex=True, **kwargs):
        '''
        :param indexer: global indexer, index nodes in dataset level.
        :param reindex: local indexer, index nodes for each graph.
        '''
        super(NetworkxGraphs2PygHeteroGraphPipeline, self).__init__(**kwargs)
        self.context_state['indexer_dict'] = {'user': Indexer(zero_start=False),
                                              'time': Indexer(zero_start=False),
                                              }
        self._user_indexer = user_indexer
        self.reindex = reindex

    @property
    def user_indexer(self):
        if isinstance(self._user_indexer, str):
            with open(self._user_indexer, 'rb') as f:
                self._user_indexer = pickle.load(f)
        return self._user_indexer

    def process(self, item=None, state=None, dataset=None, *args, **kwargs) -> Any:
        # indexer_dict = state['indexer_dict']
        indexer_dict = self.user_indexer
        hg = op.heterogeneous_graph_from_networkx(item, indexer_dict=indexer_dict, reindex=self.reindex)
        return hg


class AddTimeNodesToPygHeteroGraph(pipe.Pipeline):
    '''
    Add time nodes to heterogeneous graph.
    '''

    # added_fields = [
    #     'num_sub_cascades',
    #                 'total_time_nodes',
    #                 'time_node_indexer'
    # ]
    def __init__(self, num_time_nodes=10, sub_cascade_level=False,
                 soft_partition: Optional[int] = None,
                 **kwargs):
        '''
        :param num_time_nodes: total number of time nodes to add.
        :param sub_cascade_level: add time node in sub-cascade level or graph level,
            if True, the final number of time nodes is number_of_subcascade * num_time_nodes;
        '''
        super(AddTimeNodesToPygHeteroGraph, self).__init__(**kwargs)
        self.soft_partition = min(soft_partition, num_time_nodes - 1) if soft_partition > 0 else 0
        self.num_time_nodes = num_time_nodes

        self.context_state['num_sub_cascades'] = []
        self.context_state['time_node_indexer'] = Indexer(zero_start=False)
        self.context_state['total_time_nodes'] = 0

        self.sub_cascade_level = sub_cascade_level
        if self.num_time_nodes == 0:
            self.enable = False

    @property
    def filename_tag(self):
        tag = super(AddTimeNodesToPygHeteroGraph, self).filename_tag
        tag += f'Tm{self.num_time_nodes}'
        if self.soft_partition:
            tag += f'Sf{self.soft_partition}'
        if self.sub_cascade_level:
            tag += 'Sb'
        return tag

    def process(self, item=None, state=None, dataset=None, *args, **kwargs) -> Any:
        u_t = ('user', 'postat', 'time')
        t_t = ('time', 'pastto', 'time')
        t_u = ('time', 'contain', 'user')
        raw_item = kwargs.get('raw_item', None)
        raw_item_idx = kwargs.get('raw_item_idx', 0)
        if isinstance(item, dict):
            for k in item.keys():
                # set attribute of raw_item
                item[k].graph['raw_item'] = raw_item
            repost_graph = item[('user', 'repost', 'user')]
            g_dict = {
                # u_t: nx.DiGraph(raw_item=raw_item),
                # t_t: nx.DiGraph(raw_item=raw_item),
                # t_u: nx.DiGraph(raw_item=raw_item),
                u_t: nx.DiGraph(),
                t_t: nx.DiGraph(),
                t_u: nx.DiGraph(),
            }
            sub_cascades = op.sub_cascades(repost_graph)
            state['num_sub_cascades'].append(len(sub_cascades))

            if self.sub_cascade_level:
                graphs = [repost_graph.subgraph(cas_nodes) for cas_nodes in sub_cascades]
            else:
                graphs = [repost_graph]
            for i, g in enumerate(graphs):
                nx.set_node_attributes(g, i, 'sub_cascade_id')
                self.add_into_networkx_dict(g, g_dict, sub_cascade_id=i, state=state, raw_item_idx=raw_item_idx)
            item.update(g_dict)
        else:
            item = self.add_into_hetero_data(item)
        return item

    def add_into_networkx_dict(self, graph: nx.DiGraph, g_dict, sub_cascade_id=0, state=None, raw_item_idx=0):
        time_node_shift = self.num_time_nodes * sub_cascade_id
        time = list(nx.get_edge_attributes(graph, 'time').values())
        num_repost = graph.number_of_edges()
        linspace = np.linspace(min(time), max(time), self.num_time_nodes + 1, dtype=float).tolist()
        u_t = ('user', 'postat', 'time')
        t_u = ('time', 'contain', 'user')
        t_t = ('time', 'pastto', 'time')

        # time node - time node
        for time_node_idx in range(self.num_time_nodes):
            # add relative id (i.e.: 0 to num_time_nodes-1) to time node for each sub cascade
            state['total_time_nodes'] += 1
            # tid = state['time_node_indexer'](state['total_time_nodes'])
            tid = self.num_time_nodes * raw_item_idx + time_node_idx
            g_dict[t_t].add_node(time_node_idx + time_node_shift,
                                 sub_cascade_id=sub_cascade_id, time_node_id=time_node_idx, node_id=tid)
        for i in range(self.num_time_nodes - 1):
            g_dict[t_t].add_edge(time_node_shift + i, time_node_shift + i + 1)
        # time node - user node
        # split_index[i]：边列表的对于时间段的划分点，edges[split_index[i-1]] 至 edges[split_index[i]] 的边属于第 i 时间段
        # edges：按时间升序排列的【转发】边列表
        split_index = [0]
        edges = list(sorted(graph.edges(data=True), key=lambda x: x[2]['time']))
        for i, time_node in enumerate(linspace[1:]):
            sp_idx = split_index[-1]
            while edges[sp_idx][2]['time'] < time_node and sp_idx < len(edges):
                sp_idx += 1
            sp_idx = min(len(edges) - 1, sp_idx)
            split_index.append(sp_idx)

        for time_node_idx, edge_idx in enumerate(split_index[1:]):
            if time_node_idx and edge_idx == split_index[time_node_idx]:
                continue
            '''
            min(time_node_idx, 1): if not first element, index += 1,
              is a left open right closed section.
            '''
            # 初始化时间节点所连接的边在 edges 中的区间。
            l = split_index[time_node_idx] + min(time_node_idx, 1)
            r = edge_idx + 1
            # 针对 edges[l:r] 中的每一条边，连接软/硬划分对应的时间节点
            for edge_id in range(l, r):
                u, v, attr = edges[edge_id]
                # 硬划分对应时间节点，双向边
                time_split_id = time_node_idx + time_node_shift
                g_dict[t_u].add_edge(time_split_id, v, time=attr['time'])
                g_dict[u_t].add_edge(v, time_split_id, time=attr['time'])
                # 软划分，双向边
                for soft_shift in range(1, self.soft_partition + 1):
                    l_time, r_time = time_split_id - soft_shift, time_split_id + soft_shift
                    # 左软划分，双向边
                    if l_time >= 0:
                        g_dict[t_u].add_edge(l_time, v, time=attr['time'])
                        g_dict[u_t].add_edge(v, l_time, time=attr['time'])
                    # 右软划分，双向边
                    if r_time <= self.num_time_nodes - 1:
                        g_dict[t_u].add_edge(r_time, v, time=attr['time'])
                        g_dict[u_t].add_edge(v, r_time, time=attr['time'])
        return g_dict

    def add_into_hetero_data(self, graph: HeteroData, state=None):
        # todo: sb. to help me implement this function?
        return graph


class ObservationPipeline(pipe.Pipeline):
    '''
    Truncate DataFrame by observation time.
    '''

    def __init__(self, observation=2, **kwargs):
        if observation % 1 == 0:
            observation = int(observation)
        self.observation = observation
        self._time_delta = None
        super(ObservationPipeline, self).__init__(**kwargs)

    @property
    def filename_tag(self):
        tag = super(ObservationPipeline, self).filename_tag
        tag += f'Ob{self.observation}'
        return tag

    @property
    def time_delta(self):
        if self._time_delta is None:
            self._time_delta = timedelta(hours=self.observation)
        return self._time_delta

    def process(self, item=None, state=None, dataset=None, *args, **kwargs) -> Any:
        start = item.loc[0, 'created_at']
        end = start + self.time_delta
        rst = item[item['created_at'] <= end]
        return rst


'''
Indexer
'''


class IndexDataframePipeline(pipe.Pipeline):

    def __init__(self, indexer: Indexer, **kwargs) -> None:
        super(IndexDataframePipeline, self).__init__(**kwargs)
        self.indexer = indexer

    def process(self, item: pd.DataFrame = None, state=None, dataset=None, *args, **kwargs) -> Any:
        item['origin_uid'] = item.loc[:, 'origin_uid'].map(self.indexer)
        item['uid'] = item.loc[:, 'uid'].map(self.indexer)
        return item


class IndexEdgeListPipeline(pipe.Pipeline):

    def __init__(self, indexer: Indexer, multi_edge_list=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.indexer = indexer
        self.multi_edge_list = multi_edge_list

    def process(self, item=None, state=None, dataset=None, *args, **kwargs) -> Any:
        if not self.multi_edge_list:
            item = [item]
        for edgelist in item:
            new_list = []
            for edge in edgelist:
                e = []
                for i, node in enumerate(edge):
                    if i < 2:
                        node = self.indexer(node)
                    e.append(node)
            new_list.append(e)
        return new_list


class IndexHeteroGraphPipeline(pipe.Pipeline):
    def __init__(self, indexer_dict: Optional[Dict[str, Indexer]] = None,
                 nodes_to_index: Optional[Dict[str, bool]] = None,
                 edges_to_index: Optional[Dict[Tuple, bool]] = None,
                 **kwargs):
        super(IndexHeteroGraphPipeline, self).__init__(**kwargs)
        self.nodes_to_index = nodes_to_index
        self.edges_to_index = edges_to_index
        if indexer_dict is None:
            self.indexer_dict = None
        else:
            # = indexer_dict
            self.indexer_dict = self.reset_indexer(indexer_dict)

    def reset_indexer(self, indexer_dict):
        if self.nodes_to_index:
            for ntype, val in self.nodes_to_index.items():
                indexer_dict[ntype] = Indexer(zero_start=False) if val else None
        if self.edges_to_index:
            for etype, val in self.edges_to_index.items():
                indexer_dict[etype] = Indexer(zero_start=False) if val else None
        return indexer_dict

    def process(self, item: HeteroData, state, dataset, *args, **kwargs) -> Any:
        indexer_dict = state.get('indexer_dict', self.indexer_dict)
        # todo 异构图节点index
        for ntype in item.node_types:
            nid = item[ntype]['node_id'].numpy().tolist()
            new_nid = indexer_dict[ntype].index_items(nid)
            item[ntype]['node_id'] = torch.tensor(new_nid, dtype=torch.long)
        return item


class ExtractLabelPipeline(pipe.Pipeline):
    '''
    Extract total DataFrame length as label.
    '''

    # added_fields = ['labels']

    def __init__(self, unsqueeze=False, as_tensor=False, **kwargs):
        super(ExtractLabelPipeline, self).__init__(**kwargs)
        self.context_state['labels'] = []
        self.unsqueeze = unsqueeze
        self.as_tensor = as_tensor

    def process(self, item=None, state=None, dataset=None, *args, **kwargs) -> Any:
        pop = len(set(item['uid']))
        l = [pop] if self.unsqueeze else pop
        if self.as_tensor:
            l = torch.tensor(l, dtype=torch.float)
        state['labels'].append(l)
        return item


'''
Persist Pipeline（意义不明 orz）
'''


class PersistPipeline(pipe.Pipeline):
    # added_fields = ['number_of_processed_items']

    def __init__(self, **kwargs):
        super(PersistPipeline, self).__init__(**kwargs)
        self.context_state['number_of_processed_items'] = 0

    def process(self, item=None, state=None, dataset=None, *args, **kwargs) -> Any:
        state['number_of_processed_items'] += 1


class InmemoryPersistPipeline(PersistPipeline):
    '''
    单个文件保存所有数据，读取时一次性加载所有数据到内存
    '''

    def __init__(self, target: list, **kwargs):
        super(InmemoryPersistPipeline, self).__init__(**kwargs)
        self.target = target

    @property
    def filename_tag(self):
        tag = super(InmemoryPersistPipeline, self).filename_tag
        tag += 'memory'
        return tag

    def process(self, item, state, dataset, *args, **kwargs) -> Any:
        super(InmemoryPersistPipeline, self).process(item, state, dataset, *args, **kwargs)
        self.target.append(item)
        return item


class StorePersistPipeline(PersistPipeline):
    '''
    数据集保存到对应目录中，每个数据一个单独文件，读取时一次只读取单个数据文件
    '''

    def __init__(self, target_dir='items', filename_format='{raw_item_idx}.pkl', **kwargs):
        super(StorePersistPipeline, self).__init__(**kwargs)
        self.target_dir = osp.join(target_dir)
        self.filename_format = filename_format
        if not osp.exists(target_dir):
            os.mkdir(target_dir)

    @property
    def filename_tag(self):
        tag = super(StorePersistPipeline, self).filename_tag
        tag += 'store'
        return tag

    def process(self, item, state, dataset, *args, **kwargs) -> Any:
        super(StorePersistPipeline, self).process(item, state, dataset, *args, **kwargs)
        raw_item_idx = args[0]
        file_path = self.persist_path(raw_item_idx)
        with open(file_path, 'wb') as f:
            pickle.dump(item, f)
        return item

    def persist_path(self, raw_item_idx):
        file_name = self.filename_format.format(raw_item_idx=raw_item_idx)
        file_path = osp.join(self.target_dir, file_name)
        return file_path
