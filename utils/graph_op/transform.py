from collections import defaultdict

import torch
from torch_geometric.data import HeteroData
import torch_geometric as pyg
from torch_geometric.utils import to_networkx

from utils import Indexer
from utils import graph_op as op
import networkx as nx


def heterogeneous_graph_from_networkx(_map: dict, indexer_dict=None, reindex=True):
    g = HeteroData()
    nodes = {}
    if isinstance(indexer_dict, Indexer):
        indexer_dict = {'time': lambda x: x, 'user': indexer_dict}
    re_indexer_dict = {} if reindex else None
    for (s, e, t), v in _map.items():
        edges = v.edges()
        sources = [i for i, d in dict(v.out_degree()).items() if d > 0]
        targets = [i for i, d in dict(v.in_degree()).items() if d > 0]
        un_known_nodes = set(v.nodes()) - set(sources) - set(targets)
        for node_type in [s, t]:
            if node_type not in re_indexer_dict.keys():
                re_indexer_dict[node_type] = Indexer(zero_start=True)
            if node_type not in nodes.keys():
                nodes[node_type] = set()
        if indexer_dict and indexer_dict[s]:
            edges = map(lambda x: (indexer_dict[s](x[0]), indexer_dict[t](x[1])), edges)
            if reindex:
                # edges = [(re_indexer_dict[s][i] if s == 'user' else i,
                #           re_indexer_dict[t][j] if t == 'user' else j)
                #          for i, j in edges]
                edges = map(lambda x: (re_indexer_dict[s](x[0]), re_indexer_dict[t](x[1])), edges)
        edges = list(edges)
        g[s, e, t].edge_index = torch.tensor(edges, dtype=torch.long).T
        nodes[s].update(sources)
        nodes[t].update(targets)
        # add edge attributes into hetero graph
        edge_attrs = op.collect_edge_data(v)
        for k, attr in edge_attrs.items():
            if len(attr.items()) == len(edges):
                g[s, e, t][k] = torch.tensor(list(attr.values()))
        # add node attributes into hetero graph
        if s == t:
            node_attrs = op.collect_node_data(v)
            for k, attr in node_attrs.items():
                if len(attr.items()) == v.number_of_nodes():
                    atr = list(attr.values())
                    if isinstance(atr[0], float) or isinstance(atr[0], int):
                        atr = torch.tensor(atr)
                    g[s][k] = atr
    for node_type, node_set in nodes.items():
        g[node_type].num_nodes = len(node_set)
        if indexer_dict and indexer_dict[node_type] and reindex:
            g[node_type]['node_id'] = torch.tensor([indexer_dict[node_type](i) for i in node_set])
    return g


def pyg_to_networkx(graph: HeteroData, return_dict=True, with_attr=False):
    '''
    @param graph: 原始的pyg异构图对象
    @param return_dict: True 返回 DiGraph 字典（每种边一个DiGraph），False 返回 MultiDiGraph 对象
    @return: DiGraph or MultiDiGraph
    '''
    edge_types = graph.edge_types
    node_types = graph.node_types
    node_indxer = Indexer()
    node_to_type = {}
    type_to_node = defaultdict(set)
    if return_dict:
        nxg = {}
    else:
        nxg = nx.MultiDiGraph()
    node_attrs = ['time', 'node_id', 'raw_id', 'time_node_id']
    edge_attrs = ['time']
    graph_attrs = ['raw_item', 'y']
    for meta_edge in edge_types:
        ut, et, vt = meta_edge
        if return_dict:
            # if with_attr:
            #     if et == 'repost':
            #         node_attrs = ['node_id', 'raw_id']
            #     elif et == 'pastto':
            #         node_attrs = ['node_id', 'time_node_id', 'sub_cascade_id']
            #     elif et == 'follow':
            #         node_attrs = ['node_id', 'raw_id']
            #         edge_attrs = None
            # else:
            #     node_attrs = None
            #     edge_attrs = None
            # subg = graph.edge_type_subgraph([tuple(meta_edge)]).to_homogeneous(node_attrs=node_attrs,)
            # nxg[meta_edge] = to_networkx(subg, node_attrs, edge_attrs, graph_attrs)
            nxg[meta_edge] = nx.DiGraph()
            g = nxg[meta_edge]  # 用于后续增加边的 networkx 图对象
        else:
            g = nxg

        edge_index = graph[et]['edge_index'].T.numpy().tolist()
        edge_index = list(map(lambda x: node_indxer.index_items([f'{ut}_{x[0]}', f'{vt}_{x[1]}']), edge_index))
        for u, v in edge_index:
            node_to_type[u] = ut
            node_to_type[v] = vt
            type_to_node[ut].add(u)
            type_to_node[vt].add(v)
            g.add_node(u, node_type=ut)
            g.add_node(v, node_type=vt)

            g.add_edges_from(edge_index, edge_type=et)

    info = {  # 其他辅助信息
        'node_types': node_types,
        'edge_types': edge_types,
        'node_indxer': node_indxer,
        'node_to_type': node_to_type,
        'type_to_node': type_to_node,
    }
    return nxg, info


if __name__ == '__main__':
    from utils.data import DataModule

    dm = DataModule(name='repost', observation=2, sample=0.01, root='/root/hif/data')
    dm.prepare_data()
    ds = dm.dataset
    data = ds[0]
    print(data)
    nxg, info = pyg_to_networkx(data)
    print(nxg, info)
