import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor, wait
from functools import partial
from os import path as osp

from torch.utils.data.dataset import T_co
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from utils import Indexer
from utils.data import pipe, PipelineDataset


class CSVHeteroGraphDataset(PipelineDataset):
    # _saved_fields = {'indexer_dict'}

    def __init__(self, root, name=None, transform=None, force_reload=False, persist='memory',
                 sample=0.1,
                 **pipe_factory,
                 ):
        self.sample = int(sample) if sample % 1 == 0 else sample
        self.indexer_dict = {'user': Indexer(zero_start=False),
                             'time': Indexer(zero_start=False),
                             }
        follow_edge_path = osp.join(root, 'raw', 'global', f'{name}_relationships.txt')
        pipe_factory = pipe_factory.copy()
        pipe_factory['follow_path'] = follow_edge_path
        # pipe_factory['indexer_dict'] = self.indexer_dict
        pipe_factory['user_indexer'] = osp.join(root, 'raw', 'global', f'{name}_user_indexer.pkl')
        pipes = [
            pipe.CSV2DataframePipeline,
            pipe.LengthFilter,
            pipe.ExtractLabelPipeline,
            pipe.ObservationPipeline,
            pipe.Dataframe2SequencePipeline,
            pipe.FollowerSamplePipeline,
            pipe.AddTimeNodesToPygHeteroGraph,
            pipe.NetworkxGraphs2PygHeteroGraphPipeline,
            pipe.FollowershipFilter,
            pipe.IndexHeteroGraphPipeline
        ]

        # self.pipelines = pipes + self.pipelines
        super().__init__(root, name, pipelines=pipes, force_reload=force_reload, transform=transform,
                         persist=persist, **pipe_factory)

    @property
    def total_nodes(self):
        # 各个类型的节点数，使用 num_users 会冲突
        cnt = {node_type: len(indexer.id2item) for node_type, indexer in self.context_state['indexer_dict'].items() if
               isinstance(node_type, str)}
        # cnt = len(self.indexer_dict.id2item)
        return cnt

    @property
    def total_edges(self):
        # todo 实现统计各类型边数的功能，并优化边id等功能。
        cnt = {edge_type: len(indexer.id2item) for edge_type, indexer in self.context_state['indexer_dict'].items() if
               isinstance(edge_type, tuple)}
        return cnt

    def read_raw_items(self):
        file_list = list(filter(lambda x: x.endswith('csv'), os.listdir(self.raw_dir)))
        tot = len(file_list)
        cnt = int(tot * self.sample)
        file_list = random.sample(file_list, cnt)
        self._raw_items = list(file_list)
        super().read_raw_items()

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw', f'{self.name}_cascades')

    @property
    def meta_edges(self):
        element = self[0]
        return element.edge_types

    @property
    def save_name_tags(self):
        tags = super().save_name_tags + [f'Sp{self.sample}']
        return tags

    @property
    def save_name_suffix(self):
        return '.hg' + super().save_name_suffix

    @property
    def data_fields_from_pipeline(self):
        '''
        key: field name in Data item.
        values: field name in dataset instance, added by pipelines.
        :return:
        '''
        return {
            'y': 'labels',
            'raw_item': 'raw_items',
        }

    def sync_field_to_data(self):
        for i, data in enumerate(self.processed_items):
            for f1, f2 in self.data_fields_from_pipeline.items():
                data.__setattr__(f1, self.context_state[f2][i])

    def process(self):
        self.index_users(root=self.root, name=self.name)
        super(CSVHeteroGraphDataset, self).process()
        self.sync_field_to_data()

    def index_users(self, root, name, reindex=False):

        global_dir = osp.join(root, 'raw', 'global')
        indexer_path = osp.join(global_dir, f'{name}_user_indexer.pkl')
        if not osp.exists(indexer_path) or reindex:
            print('\nIndex all users: processing.', end='')
            indexer = Indexer(zero_start=False)
            print(f'\rIndex all users: {name}_relationships.txt', end='')
            with open(osp.join(global_dir, f'{name}_relationships.txt'), 'r') as f:
                for line in f.readlines():
                    u, v = line.strip().split(',')
                    u, v = int(u), int(v)
                    # u, v = int(u,16), int(v,16)
                    indexer.index_items([u, v])
            print(f'\rIndex all users: {self.raw_dir}', end='')
            for raw_item in self.raw_items:
                p = osp.join(self.raw_dir, raw_item)
                import pandas as pd
                df = pd.read_csv(p, encoding='utf8')
                df['origin_uid'] = df['origin_uid'].fillna(df['uid'])
                indexer.index_items(df['uid'].astype(int).to_list())
                # indexer.index_items(df['uid'].apply(lambda x: int(x, base=16)).astype(int).to_list())
                indexer.index_items(df['origin_uid'].astype(int).to_list())
                # indexer.index_items(df['origin_uid'].apply(lambda x: int(x, base=16)).astype(int).to_list())
            with open(indexer_path, 'wb') as f:
                pickle.dump(indexer, f)
            print('\rIndex all users: done.')

    def __getitem__(self, index) -> T_co:
        item = super(CSVHeteroGraphDataset, self).__getitem__(index=index)
        # label = self.labels[index]
        return item


if __name__ == '__main__':
    dataset_pipe_factory = {'root': '/root/hif/data',
                            'name': 'twitter', 'persist': 'memory',
                            'observation': 1, 'num_time_nodes': 6, 'soft_partition': 2,
                            # 'max_length': 500,
                            'min_length': 20,
                            'method': 'dy', 'hop': 1,
                            'force_reload': 1, 'sample': 0.05,
                            'alpha': 10, 'beta': 200,
                            'source_base': 6, 'reposted_base': 4, 'leaf_base': 5, 'cas_base': 6,
                            'sample_batch': 200,
                            'max_followerships': 2e4,
                            # transform:[pipe.AddTimeNodesToPygHeteroGraph()],
                            }
    # 提前处理多个数据集
    fus = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for n in ['MSC',]:
            for ob in [0.5, 1]:
                for me in ['dy', 'hop']:
                    d2 = {'name': n, 'observation': ob, 'sample': 1 if n == 'twitter' else 0.1,
                          'method': me}
                    d = dataset_pipe_factory.copy()
                    d.update(d2)
                    fu = pool.submit(CSVHeteroGraphDataset, **d)
                    cb = partial(print, d)
                    fu.add_done_callback(cb)
                    fus.append(fu)
    wait(fus)

    # dataset = CSVHeteroGraphDataset(num_workers=0, **dataset_pipe_factory)
    # # test for parallel run and persist
    # with open('test.pkl', 'wb') as f:
    #     pickle.dump(dataset, f)
    # with open('test.pkl', 'rb') as f:
    #     ds_load = pickle.load(f)
    #     print('pickle load:\n', ds_load)
    #
    # dataloader = DataLoader(dataset, batch_size=2, num_workers=2)
    # print(dataset)
    # print(dataset.save_name)
    # print(dataset.meta_edges)
    # import pandas as pd
    #
    # df = pd.DataFrame([
    #     {'y': data.y,
    #      'users': data.num_nodes_dict['user'],
    #      'followerships': data.num_edges_dict[('user', 'follow', 'user')],
    #      'reposts': data.num_edges_dict[('user', 'repost', 'user')] }
    #     for data in tqdm(dataset)
    #     ])
    #
    # for i, data in enumerate(dataset):
    #     print(i, data)
    # break
    # data = dataset[i]
    # df = df.append({
    #     'y': data.y,
    #     'users': data.num_nodes_dict['user'],
    #     'followerships': data.num_edges_dict[('user', 'follow', 'user')],
    #     'reposts': data.num_edges_dict[('user', 'repost', 'user')],
    #
    # }, ignore_index=True)
    #
    #     # data = i
    #     # print(f'{i}: {data.y}')
    #     # print('label:', data.y)
    #     # print(data)
    #     # print(data.metadata())
    #     # break
    # print(df.describe())
    # print(df.head(15))
