import math
from argparse import ArgumentParser
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler

from utils.data import PipelineDataset
from utils.data.datasets import CSVHeteroGraphDataset

__all__ = ['DataModule']


class DataModule(pl.LightningDataModule):

    def __init__(self, batch_size=32, split_rate=None, collate_fn=None, num_workers=8,
                 dataset_cls=CSVHeteroGraphDataset, dataloader_cls=DataLoader,
                 # train_transforms=None, val_transforms=None, test_transforms=None, dims=None,
                 drop_last=False,
                 **kwargs):
        if split_rate is None:
            split_rate = [0.7, 0.15, 0.15]
        self.batch_size = batch_size
        self.dataset_cls = dataset_cls
        self.dataloader_cls = dataloader_cls
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.dataset_kwargs = kwargs
        self.num_workers = num_workers
        self._dataset = None
        self.train_set = None
        self.valid_set = None
        self.test_set = None
        self.split_rate = split_rate
        super().__init__()
        self.save_hyperparameters('batch_size')

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = parent_parser.add_argument_group('DataModule')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--observation', type=float, default=2)
        parser.add_argument('--min_length', type=int, default=20, help='min length of cascades')
        parser.add_argument('--max_length', type=int, default=None, help='max length of cascades')
        parser.add_argument('--alpha', type=int, default=80, help='weight to adjust sampled follower.')
        parser.add_argument('--beta', type=int, default=0, help='minimum number of sampled follower.')
        parser.add_argument('--hop', type=int, default=1, help='hops of sampled follower.')
        parser.add_argument('--sample_batch', type=int, default=15, help='batch size of follower sampling.')
        parser.add_argument('--soft_partition', type=int, default=0, help='time node soft partition size.')
        parser.add_argument('--sample', type=float, default=0.1)
        parser.add_argument('--reverse', action='store_true', default=False, help='reverse sampled follower network')
        parser.add_argument('--source_base', type=float, default=3, help='weight of source nodes')
        parser.add_argument('--reposted_base', type=float, default=1.5, help='weight of reposted nodes')
        parser.add_argument('--leaf_base', type=float, default=2, help='weight of leaf nodes')
        parser.add_argument('--cas_base', type=float, default=4, help='weight of sub-cascade')
        parser.add_argument('--root', type=str, default='/root/hif/data')
        parser.add_argument('--name', type=str, default='repost')
        parser.add_argument('--max_followerships', default=None,
                            help='max number of follower-ships, '
                            'None or integers and support exponential format e.g. 1e5')
        parser.add_argument('--method', type=str, default='bfs', choices=['bfs', 'hop', 'dy'],
                            help='method to sample follower.')
        parser.add_argument('--force_reload', default=False, action='store_true')
        parser.add_argument('--relative_timestamp', default=False, action='store_true')
        parser.add_argument('--drop_last', default=False, action='store_true')
        parser.add_argument('--sub_cascade_level', action='store_true', default=False)
        return parent_parser

    def data_args(self, args=None):
        '''
        args：待获取的参数名列表，为None时默认使用所有需要记录的（会影响模型最终结果的）参数（部分 add_argparse_args 中涉及的参数）
        @return: 返回对应的数据集参数字典
        '''
        if args is None:
            args = [
                # base args:
                'name', 'batch_size', 'observation', 'sample', 'drop_last', 'min_length', 'max_length',
                # args for follower sampling:
                'alpha', 'beta', 'method', 'max_followerships',
                'hop', 'source_base', 'reposted_base', 'leaf_base',
                # args for adding time nodes:
                'soft_partition', 'relative_timestamp', 'sub_cascade_level', 'num_time_nodes',
            ]
        arg_dict = {}
        for arg in args:
            if arg in vars(self).keys():
                arg_dict[arg] = getattr(self, arg)
            else:
                arg_dict[arg] = self.dataset_kwargs[arg]
        name_id = {'SSC': 0, 'MSC': 1, 'twitter': 2}
        arg_dict['name_id'] = name_id[arg_dict['name']]
        # arg_dict['name_id'] = name_id[self.name]

        return arg_dict

    @property
    def dataset(self) -> PipelineDataset:
        if not self._dataset:
            self.prepare_data()
        return self._dataset

    def prepare_data(self) -> None:
        self._dataset = self.dataset_cls(**self.dataset_kwargs)
        pl.seed_everything()
        super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        tot = len(self._dataset)
        cnt = [int(tot * r) for r in self.split_rate[:-1]]
        cnt.append(tot - sum(cnt))
        self.train_set, self.valid_set, self.test_set = random_split(self._dataset, cnt)
        super().setup(stage)

    def weighted_sampler(self, dataset, replacement=True, smoothing=0.7):
        weights = [data.y for data in dataset]
        total = sum(weights)
        n = len(weights)
        a = (1 - smoothing)
        b = smoothing / n
        smoothed_weights = list(map(lambda x: x / total * a + b, weights))
        sampler = WeightedRandomSampler(weights=smoothed_weights,
                                        num_samples=int(0.7*n), replacement=replacement)
        return sampler

    def dataloader_kwargs(self, stage='train'):
        kwargs = {
            'dataset': getattr(self, f'{stage}_set'),
            'batch_size': self.batch_size,
            'shuffle': True if stage == 'train' else False,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
        }
        # if self.dataloader_cls == DataLoader:
        #     kwargs['num_workers'] = self.num_workers
        return kwargs

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        loader = self.dataloader_cls(dataset=self.train_set, batch_size=self.batch_size,
                                     # shuffle=True,
                                     sampler=self.weighted_sampler(self.train_set),
                                     num_workers=self.num_workers, collate_fn=self.collate_fn,
                                     pin_memory=True,
                                     # prefetch_factor=4,
                                     drop_last=self.drop_last)
        return loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        loader = self.dataloader_cls(dataset=self.valid_set, batch_size=self.batch_size, num_workers=self.num_workers,
                                     pin_memory=True,
                                     # prefetch_factor=4,
                                     collate_fn=self.collate_fn, drop_last=self.drop_last)
        return loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        loader = self.dataloader_cls(dataset=self.test_set, batch_size=self.batch_size, num_workers=self.num_workers,
                                     pin_memory=True,
                                     # prefetch_factor=4,
                                     collate_fn=self.collate_fn, drop_last=self.drop_last)
        return loader
