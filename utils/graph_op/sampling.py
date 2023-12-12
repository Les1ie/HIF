import math
import random
from collections import deque, defaultdict
from functools import partial, reduce
from operator import itemgetter
from queue import PriorityQueue

import networkx as nx

from utils import graph_op as op


def multi_source_bfs(g, source_nodes, max_nodes) -> list:
    vis = set()
    source_nodes = filter(lambda x: x in g.nodes(), source_nodes)
    queues = list(deque([i]) for i in source_nodes)
    has_new_node = True
    while has_new_node and len(vis) < max_nodes:
        has_new_node = False
        for i, q in enumerate(queues):
            if not len(q):
                continue
            head = q.popleft()
            if head in vis:
                continue
            vis.add(head)
            has_new_node = True
            for nei in g[head]:
                if nei not in vis:
                    q.append(nei)
            if len(vis) >= max_nodes:
                break

    return list(vis)


def weighted_bfs(repost_g: nx.DiGraph, follow_g: nx.DiGraph, ud_follow_g: nx.Graph,
                 alpha=80, beta=100, sample_batch=15, source_base=3, reposted_base=1.5, leaf_base=2):
    f'''
    动态带权BFS游走采样，对候选采样起点分配动态更新的权重，每次挑选权重最大的节点作为起点进行一轮采样，
      采样 {sample_batch} 数量的节点后更新权重，再进行下一轮采样，直至无新节点或达到总节点数限制。
    总采样节点数计算为 {alpha} * N + {beta}，其中 N 为 {repost_g.number_of_nodes()}。
    初始以及更新的节点权重计算为：base + lg(1+d) - lg(1+n)，其中：
      base 为对应节点位置的权重（对应 {source_base}、{reposted_base}、{leaf_base}），
      d 为对应节点在 {follow_g} 中的出度，
      n 以该节点为起点为已采样的节点数。
    @param repost_g: 转发图，仅使用其中的节点集合作为游走的起点
    @param follow_g: 有向关注图，用于采样的全局图
    @param ud_follow_g: 无向的关注图，通过外层传入对象防止每次采样都对 {follow_g} 进行无向处理，提高性能
    @param alpha: 转发节点数的对数系数
    @param beta: 采样的节点数量下限基数
    @param sample_batch: 每一轮采样最多采样的节点数
    @param source_base: 源节点位置权重
    @param reposted_base: 中间节点位置权重
    @param leaf_base: 叶子节点位置权重
    @return: 采样到的节点集合
    '''
    num_repost_nodes = repost_g.number_of_nodes()
    max_nodes = beta + math.ceil(alpha * math.log(1 + num_repost_nodes))
    source_nodes = op.source_nodes(repost_g)
    reposted_nodes = op.influenced_nodes(repost_g)
    leaf_nodes = op.leaf_nodes(repost_g)
    # todo 考虑使用基于权重的随机选择（无法处理负数权重），但是能防止“饥饿”等待现象
    queue = PriorityQueue()
    sampled_nodes = set()
    bfs_iters = dict()
    sampled_nodes_dict = defaultdict(set)
    node_bases = [(source_nodes, source_base), (reposted_nodes, reposted_base), (leaf_nodes, leaf_base)]

    def node_weight(n):
        weight = 0
        for node_set, base in node_bases:
            if n in node_set:
                weight = base
                break
        degree = 0 if n not in follow_g.nodes() else follow_g.out_degree(n)
        # weight += remain_nodes[n] / degree * math.log(1 + degree)
        weight += math.log(1 + degree) - math.log(1 + len(sampled_nodes_dict[n]))
        return -weight

    for n in repost_g.nodes():
        bfs_iters[n] = None if n not in follow_g.nodes() else nx.bfs_edges(ud_follow_g, n)
        w = node_weight(n)
        queue.put((w, n))

    while len(sampled_nodes) < max_nodes and queue.qsize():
        w, n = queue.get()
        bfs_nodes = set()
        bfs_nodes.add(n)
        while len(bfs_nodes) <= sample_batch and bfs_iters[n] is not None and len(bfs_nodes) + len(
                sampled_nodes) <= max_nodes:
            try:
                u, v = next(bfs_iters[n])
                if v not in sampled_nodes_dict[n]:
                    bfs_nodes.add(v)
            except Exception:
                bfs_iters[n] = None
                break
        sampled_nodes_dict[n].update(bfs_nodes)
        sampled_nodes.update(bfs_nodes)
        if bfs_iters[n] is not None:
            # None 表示该节点已没有可再通过BFS游走得到的节点
            w = node_weight(n)
            queue.put((w, n))
    sampled_nodes |= repost_g.nodes
    return sampled_nodes


def dynamic_weighted_sampling(repost_g: nx.DiGraph, follow_g: nx.DiGraph,
                              alpha: int = 10, beta: int = 100, sample_batch: int = 50,
                              source_base=5, reposted_base=3, leaf_base=4, cas_base=4) -> set:
    reversed_f = nx.reverse_view(follow_g)
    follow_g_nodes = set(follow_g.nodes())
    repost_g_nodes = set(repost_g.nodes())
    sampled_nodes = set()
    total = alpha * repost_g.number_of_nodes() + beta
    source_nodes = op.source_nodes(repost_g)
    reposted_nodes = op.influenced_nodes(repost_g, False)
    leaf_nodes = op.leaf_nodes(repost_g)
    sub_cascades = op.sub_cascades(repost_g)

    reposter_based_weight = {u: source_base if u in source_nodes else leaf_base if u in leaf_nodes else reposted_base
                             for u in repost_g.nodes}
    # follower_of_reposter = {u: set(reversed_f[u]) for u in repost_g.nodes if u in reversed_f.nodes}
    follower_of_reposter = defaultdict(set)
    for u in repost_g.nodes:
        if u in reversed_f.nodes:
            follower_of_reposter[u] = set(reversed_f[u])
    candidate_nodes = reduce((lambda a, b: a | b), follower_of_reposter.values(), set()) - set(repost_g.nodes)

    if len(candidate_nodes) <= total:
        return candidate_nodes

    def update_reposter_weight():
        weight_dict = defaultdict(float)
        for u in repost_g.nodes:
            num_f = len(follower_of_reposter.get(u, []))
            if num_f:
                w = reposter_based_weight.get(u, 0) + math.log(num_f)
                weight_dict[u] = w
        return weight_dict
    potential_followers = set()
    def update_follower_weight(rw):
        weight_dict = defaultdict(float)
        for cas in sub_cascades:
            cas_followers = set()
            for u in cas:
                cas_followers.update(follower_of_reposter[u])
            for v in cas_followers:
                if weight_dict[v] > 0:
                    potential_followers.add(v)
                weight_dict[v] += cas_base
                for u in follow_g[v]:
                    weight_dict[v] += rw.get(u, 0)
        return weight_dict

    # sampling
    while len(sampled_nodes) < total and len(candidate_nodes):
        step_max = min(len(candidate_nodes), sample_batch, total - len(sampled_nodes))
        fw = update_follower_weight(update_reposter_weight())
        sorted_followers = map(itemgetter(0), sorted(fw.items(), key=itemgetter(1), reverse=True))
        for i, v in enumerate(sorted_followers):
            # v 关注者
            if i >= step_max:
                break

            sampled_nodes.add(v)
            # if v in follow_g.nodes():
            for u in follow_g[v]:
                follower_of_reposter[u].discard(v)
            candidate_nodes.discard(v)
    sampled_nodes |= repost_g_nodes
    return sampled_nodes


def k_hop(repost_g: nx.MultiDiGraph, ud_follow_g: nx.Graph,
          hop: int = 1, max_follower=None):
    # ud_fg = ud_follow_g.to_undirected(as_view=True)
    base_nodes = set(repost_g.nodes())
    outer_nodes = base_nodes.copy()
    sampled_nodes = base_nodes.copy()
    for h in range(hop):
        nei = set()
        for n in outer_nodes:
            if n in ud_follow_g:
                nei.update(nx.neighbors(ud_follow_g, n))
        nei -= sampled_nodes
        sampled_nodes.update(nei)
        outer_nodes = nei
    return sampled_nodes
