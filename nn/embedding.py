__all__ = ['BaseEmbeddings', 'LearnableEmbeddings', 'CollatedEmbeddings']

from typing import Optional, Dict, Union

from torch import nn

NUM_EMBEDDINGS = Union[int, Dict[str, int]]
EMBEDDINGS = Union[nn.Embedding, nn.ModuleDict]


class BaseEmbeddings(nn.Module):
    def __init__(self,
                 total_nodes: Optional[NUM_EMBEDDINGS] = None,
                 total_edges: Optional[NUM_EMBEDDINGS] = None,
                 *args, **kwargs):
        super().__init__()
        self.total_nodes = total_nodes
        self.total_edges = total_edges

    def get_node_embeddings(self, *args, **kwargs) -> Optional[EMBEDDINGS]:
        return None

    def get_edge_embeddings(self, *args, **kwargs) -> Optional[EMBEDDINGS]:
        return None


class LearnableEmbeddings(BaseEmbeddings):
    '''
    基于 torch.nn.Embedding 的可学习初始嵌入表
    '''
    def __init__(self, *args, **kwargs):
        super(LearnableEmbeddings, self).__init__(*args, **kwargs)
        if not hasattr(self, 'in_feats'):
            self.in_feats = kwargs['in_feats']
        self.init_node_embeddings = self.get_init_embeddings(self.total_nodes)
        self.init_edge_embeddings = self.get_init_embeddings(self.total_edges)

    def get_init_embeddings(self, num_dict: NUM_EMBEDDINGS) -> Optional[EMBEDDINGS]:
        embeddings = None
        if isinstance(num_dict, dict):
            # heterogeneous graph embeddings
            embeddings = nn.ModuleDict(
                {k: nn.Embedding(v, self.in_feats) for k, v in num_dict.items()})
        elif isinstance(num_dict, int):
            # homogeneous graph embeddings
            embeddings = nn.Embedding(num_dict, self.in_feats)
        return embeddings

    def get_edge_embeddings(self, *args, **kwargs) -> Optional[EMBEDDINGS]:
        batch = kwargs.get('batch')
        if self.init_edge_embeddings is None:
            return None
        elif isinstance(self.init_edge_embeddings, nn.ModuleDict):
            emb = {edge_type: self.init_edge_embeddings[edge_type](batch[edge_type]['edge_id'])
                   for edge_type in batch.edge_types}
        else:
            emb = self.init_edge_embeddings(batch['edge_id'])
        return emb

    def get_node_embeddings(self, *args, **kwargs) -> Optional[EMBEDDINGS]:
        batch = kwargs.get('batch')
        if self.init_node_embeddings is None:
            emb = None
        elif isinstance(self.init_node_embeddings, nn.ModuleDict):
            emb = {node_type: self.init_node_embeddings[node_type](batch[node_type]['node_id'])
                   for node_type in batch.node_types}
        else:
            emb = self.init_node_embeddings(batch['node_id'])
        return emb


class CollatedEmbeddings(BaseEmbeddings):
    '''
    dataloader.collate_fn 中完成的特征生成，加载时可通过 num_workers 进行加速
    '''
    def get_node_embeddings(self, *args, **kwargs) -> Optional[EMBEDDINGS]:
        batch = kwargs['batch']
        return batch['x']['node']
    
    def get_edge_embeddings(self, *args, **kwargs) -> Optional[EMBEDDINGS]:
        batch = kwargs['batch']
        return batch['x']['edge']
