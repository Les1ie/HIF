import networkx as nx

__all__ = ['collect_edge_data', 'collect_node_data']


def collect_edge_data(g: nx.Graph):
    edges = list(g.edges(data=True))
    attrs = {}
    for s, t, attr in edges:
        for k, v in attr.items():
            if k not in attrs.keys(): attrs[k] = {}
            attrs[k][(s, t)] = v
    return attrs


def collect_node_data(g: nx.Graph, nodes=None):
    if not nodes:
        nodes = list(g.nodes())
    data = {}
    if len(nodes):
        attrs = g.nodes[nodes[0]].keys()
        for attr in attrs:
            if attr not in data.keys(): data[attr] = {}
            for n in nodes:
                data[attr][n] = g.nodes[n][attr]
    return data
