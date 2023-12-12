import networkx as nx

__all__ = ['leaf_nodes', 'source_nodes', 'influenced_nodes', 'common_follower',
           'order_by_degree', 'sub_cascades']


def leaf_nodes(g: nx.DiGraph):
    nodes_with_selfloops = list(nx.nodes_with_selfloops(g))
    leaves = set()
    for i in g.nodes():
        selfloop = int(i in nodes_with_selfloops)
        if g.in_degree(i) >= 1 + selfloop and g.out_degree(i) == 0 + selfloop:
            leaves.add(i)
    return leaves


def source_nodes(g: nx.DiGraph):
    nodes_with_selfloops = list(nx.nodes_with_selfloops(g))
    sources = set()
    for i in g.nodes:
        selfloop = int(i in nodes_with_selfloops)
        if g.in_degree(i) == 0 + selfloop:
            sources.add(i)
    return sources


def influenced_nodes(g: nx.DiGraph, contain_leafs=True):
    nodes_with_selfloops = set(nx.nodes_with_selfloops(g))
    nodes = []
    for i in g.nodes():
        selfloop = int(i in nodes_with_selfloops)
        if g.in_degree(i) >= 1 + selfloop:
            if not contain_leafs and g.out_degree(i) == selfloop:
                continue
            nodes.append(i)
    return nodes


def order_by_degree(g: nx.DiGraph, nodes, k=0):
    nodes = list(sorted(nodes, key=lambda x: g.out_degree(x) if x in g.nodes() else 0))
    if k:
        return nodes[-k:]
    else:
        return nodes


def common_follower(repost_g: nx.DiGraph, follow_g: nx.DiGraph,
                    max_hop=1) -> set:
    '''
    对于转发图中的每个子级联，找到至少关注了两个子级联中用户的关注者。
    :param repost_g: 转发图
    :param follow_g: 关注图，边 (u, v) 表示 u 关注了 v，u 是 v 的关注者
    :param max_hop: 考虑共同关注者的最大跳数，默认为 1
    :return: 满足条件的节点子集合
    '''
    sub_cas = sub_cascades(repost_g)
    num_sub_cas = len(sub_cas)
    common_followers = set()

    sub_cas_followers = []
    follow_g_nodes = set(follow_g.nodes)
    for cas in sub_cas:
        outer = cas
        nei = set(cas)
        new_nei = set()
        for h in range(max_hop):
            for i in outer:
                if i in follow_g_nodes:
                    new_nei.update(follow_g[i])
            new_nei -= nei
            nei.update(new_nei)
            outer = new_nei
        sub_cas_followers.append(nei)
    for i in range(num_sub_cas):
        for j in range(i + 1, num_sub_cas):
            common_followers.update(sub_cas_followers[i] & sub_cas_followers[j])

    return common_followers


def sub_cascades(g: nx.DiGraph):
    '''
    提取 g 中的子级联，如果给定源点集合 source_nodes 则仅返回给定源点相关的子级联
    @param g:
    @param source_nodes:
    @return:
    '''
    g = g.copy()
    g.remove_edges_from(nx.selfloop_edges(g))
    components = list(nx.connected_components(g.to_undirected(as_view=True)))
    return components
    # if source_nodes is None:
    #     # inference source node by in-degrees,
    #     # not suit for graphs that exists edges between source nodes
    #     in_degrees = dict(g.in_degree())
    #     for n in nx.nodes_with_selfloops(g):
    #         in_degrees[n] -= 1
    #     source_nodes = [n for n, d in in_degrees.items() if d == 0]
    # cascades = {
    #     n: {n} for n in source_nodes
    # }
    # for source_node in cascades.keys():
    #     cascades[source_node].update(nx.bfs_tree(g, source_node).nodes())
    # sub_cascades_nodes = list(cascades.values())
    # if not len(sub_cascades_nodes):
    #     sub_cascades_nodes.extend(g.nodes)
    # return sub_cascades_nodes
