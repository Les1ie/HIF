from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import networkx as nx

from utils import graph_op as op
from utils.data import DataModule

__all__ = ['Profile', 'draw', 'cascade_profiles',
           'draw_cascade', 'draw_hetero_graph', 'draw_pyg_hetero_graph']


class Profile:
    def __init__(self, name, judge_func=None, layout_func=nx.spring_layout, limit=None,
                 label_kwargs={}, enable=True, **kwargs):
        self.name = name
        self.draw_kwargs = kwargs
        self.label_kwargs = label_kwargs
        self.judge_func = judge_func
        self.layout_func = layout_func
        self.enable = enable
        self.limit = limit
        self.counter = 0

    def judge(self, graph, item):
        if not self.enable:
            return False
        if self.limit is not None and self.counter >= self.limit:
            return False
        if self.judge_func:
            jud = self.judge_func(graph, item)
            if jud:
                self.counter += 1
            return jud
        return True


def draw(g, node_profiles=None, edge_profiles=None, contain_data=False, ignore_default=False):
    if edge_profiles is None:
        edge_profiles = []
    if node_profiles is None:
        node_profiles = []
    node_dict = defaultdict(set)
    edge_dict = defaultdict(set)
    if not ignore_default:
        default = 'default'
        node_profiles.append(Profile(default))
        edge_profiles.append(Profile(default))
    node_profile_dict = {p.name: p for p in node_profiles}
    edge_profile_dict = {p.name: p for p in edge_profiles}
    for n in g.nodes(data=contain_data):
        for p in node_profiles:
            if p.judge(g, n):
                node_dict[p.name].add(n[0])
                break
    for e in g.edges(data=contain_data):
        for p in edge_profiles:
            if p.judge(g, e):
                edge_dict[p.name].add(e[:-1])
                break

    global_pos = {}
    for name, nodes in node_dict.items():
        p = node_profile_dict[name]
        if not p.enable: continue
        pos = p.layout_func(g, k=10)
        pos.update(global_pos)
        global_pos = pos
        nx.draw_networkx_nodes(g, pos=pos, nodelist=nodes, **p.draw_kwargs)
        if p.label_kwargs is not None:
            labels = {n: n for n in nodes}
            nx.draw_networkx_labels(g, pos=pos, labels=labels, **p.label_kwargs)
    for name, edges in edge_dict.items():
        p = edge_profile_dict[name]
        if not p.enable: continue
        pos = p.layout_func(g)
        pos.update(global_pos)
        global_pos = pos
        nx.draw_networkx_edges(g, pos=pos, edgelist=edges, **p.draw_kwargs)
        if p.label_kwargs is not None:
            labels = {e: e for e in edges}
            nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=labels, **p.label_kwargs)


def cascade_profiles():
    node_profiles = [Profile('source',
                             judge_func=lambda g, n: n[0] in op.source_nodes(g),
                             node_color='red',
                             label='source node',
                             label_kwargs={}),
                     Profile('reposted',
                             judge_func=lambda g, n: n[0] in op.influenced_nodes(g, False),
                             node_color='orange',
                             label='reposted node',
                             label_kwargs={}),
                     Profile('leaf',
                             judge_func=lambda g, n: n[0] in op.leaf_nodes(g),
                             node_color='green',
                             label='leaf node',
                             label_kwargs={}),
                     ]
    edge_profiles = [Profile('repost',
                             node_color='black',
                             label='leaf node',
                             ), ]
    return node_profiles, edge_profiles


def draw_cascade(g: nx.DiGraph):
    profiles = cascade_profiles()
    draw(g, *profiles)
    return


def draw_hetero_graph(graph: dict):
    '''
    用于分析采样算法
    @param graph:
    @return:
    '''
    follow_nodes = graph['follow'].nodes()
    sampled_nodes = graph['sampled'].nodes()
    repost_edges = graph['repost'].edges()
    follow_edges = graph['follow'].edges()
    sampled_edges = graph['sampled'].edges()
    ep = [Profile('repost', lambda g, x: x in repost_edges, edge_color='black', label='leaf node', ),
          Profile('sampled', lambda g, x: x in sampled_edges, edge_color='cyan',
                  connectionstyle='arc3,rad=0.15', alpha=0.7),
          Profile('follow', lambda g, x: x in (follow_edges - sampled_edges), edge_color='gray',
                  connectionstyle='arc3,rad=0.15', alpha=0.3), ]
    # np, ep = cascade_profiles()
    np = [Profile('source', judge_func=lambda g, n: n[0] in op.source_nodes(graph['repost']),
                  node_color='red', label='source node', label_kwargs={}),
          Profile('reposted', judge_func=lambda g, n: n[0] in op.influenced_nodes(graph['repost'], False),
                  node_color='orange', label='reposted node', label_kwargs={}),
          Profile('leaf', judge_func=lambda g, n: n[0] in op.leaf_nodes(graph['repost']),
                  node_color='green', label='leaf node', label_kwargs={}),
          Profile('sampled', lambda g, x: x[0] in sampled_nodes, node_color='cyan', label_kwargs={}),
          Profile('follower', lambda g, x: x[0] in (follow_nodes - sampled_nodes),
                  node_color='gray', label_kwargs={}),
          ]
    g = nx.MultiDiGraph()
    g.add_edges_from(repost_edges)
    g.add_edges_from(follow_edges)
    draw(g, np, ep)


def draw_pyg_hetero_graph(graphs: dict, info: dict):
    '''
    配合 utils.graph_op.pyg_to_networkx 使用
    @param graphs: 异构图字典
    @param info: 其他辅助信，由 utils.graph_op.pyg_to_networkx 返回
    @return:
    '''
    ufu = ('user', 'follow', 'user')
    uru = ('user', 'repost', 'user')
    upt = ('user', 'postat', 'time')
    tpt = ('time', 'pastto', 'time')
    tcu = ('time', 'contain', 'user')
    follow_nodes = set(graphs[ufu].nodes())
    repost_nodes = set(graphs[uru].nodes())
    repost_edges = graphs[uru].edges()
    follow_edges = graphs[ufu].edges()
    et = 'edge_type'
    nt = 'node_type'
    ep = [Profile('repost', lambda g, x: x[2][et] == 'repost', edge_color='orange', label='repost', label_kwargs=None),
          Profile('follow', lambda g, x: x[2][et] == 'follow', edge_color='gray',
                  connectionstyle='arc3,rad=0.15', alpha=0.2, label_kwargs=None),
          Profile('post at/contain', lambda g, x: x[2][et] in ['postat', 'contain'],
                  edge_color='green', label='post at/contain', label_kwargs=None),
          Profile('past to', lambda g, x: x[2][et] == 'pastto', edge_color='blue', label='past to', label_kwargs=None),
          ]
    # np, ep = cascade_profiles()
    cascade_node_size = 50
    time_node_size = cascade_node_size
    follower_node_size = 35
    np = [Profile('source', judge_func=lambda g, n: n[0] in op.source_nodes(graphs[uru]), node_size=cascade_node_size,
                  node_color='red', label='source user', label_kwargs={}),
          Profile('reposted', judge_func=lambda g, n: n[0] in op.influenced_nodes(graphs[uru], False),
                  node_size=cascade_node_size,
                  node_color='orange', label='reposted user', label_kwargs={}),
          Profile('leaf', judge_func=lambda g, n: n[0] in op.leaf_nodes(graphs[uru]), node_size=cascade_node_size,
                  node_color='green', label='leaf user', label_kwargs={}),
          Profile('follower', lambda g, x: x[0] in follow_nodes - repost_nodes, node_size=follower_node_size,
                  node_color='gray', label='follower user', alpha=0.2, label_kwargs=None),
          Profile('time', lambda g, x: x[1][nt] == 'time', node_size=time_node_size,
                  node_color='blue', label='time node', label_kwargs={}),
          ]
    g = nx.MultiDiGraph()
    for nxg in graphs.values():
        g.add_edges_from(nxg.edges(data=True))
        g.add_nodes_from(nxg.nodes(data=True))
    draw(g, np, ep)
    return


if __name__ == '__main__':
    # Test:
    # rg = nx.DiGraph()
    # rg.add_edges_from([(0, 1), (1, 2), (2, 7)])
    # rg.add_edges_from([(3, 4), (4, 5), (3, 6)])
    # # nx.draw(rg, with_labels=True), plt.show()
    # # draw_cascade(rg), plt.show()
    # fg = nx.gnm_random_graph(20, 60, directed=True)
    # # nx.draw(fg, with_labels=True), plt.show()
    # sp = op.weighted_bfs(rg, fg, bfs_batch=2, alpha=5)
    # sfg = fg.subgraph(sp)
    # # nx.draw(sfg, with_labels=True), plt.show()
    # draw_hetero_graph({'repost': rg, 'follow': fg, 'sampled': sfg}), plt.show()
    # Run
    dm = DataModule(name='repost', observation=1,
                    soft_partition=2, num_time_nodes=5,
                    alpha=20, beta=20, bfs_batch=5,
                    source_base=3, reposted_base=4, leaf_base=6,
                    force_reload=True,
                    sample=0.001, root='/root/hif/data')
    dm.prepare_data()
    ds = dm.dataset
    for data in ds:
        # data = ds[4]
        if data.num_edges_dict[('user', 'repost', 'user')] < 2 or data.num_edges_dict[('user', 'repost', 'user')] > 10:
            continue
        print(data)
        nxg, info = op.pyg_to_networkx(data)
        draw_pyg_hetero_graph(nxg, info)
        plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.1))
        plt.show()
        break
