import os
import pickle
from datetime import timedelta
from os import path as osp

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset
from tqdm.auto import tqdm

from utils import Indexer

__all__ = ['WeiboRepostDataset']


class WeiboRepostDataset(Dataset):
    name = 'WeiboRepost'
    version = 0.1

    def __init__(self, root='data/weibo_repost_01', mode='train', force_reload=False
                 , zero_start=True, observe=1, max_length=512, enhance=True, mask_rate=0.05):
        self.force_reload = force_reload
        self.zero_start = zero_start
        self.mode = mode
        self.root = root
        self.enhance = enhance
        self.mask_rate = mask_rate
        self.observe = observe
        self.max_length = max_length
        self.raw_dir = osp.join(self.root, 'raw')
        self.processed_dir = osp.join(self.root, 'processed')
        if not osp.exists(self.processed_dir):
            os.mkdir(self.processed_dir)
        self.save_path = osp.join(self.processed_dir, f'{self.mode}.{self.name}.pkl')
        self.repost_path = osp.join(self.raw_dir, f"{self.mode}.repost.csv")
        self.origin_weibo_path = osp.join(self.raw_dir, f"{self.mode}.origin_weibo.csv")
        self.user_profile_path = osp.join(self.raw_dir, "user_profile.csv")

        self._user_profile = None
        self._origin_weibo = None
        self._repost = None

        self.user_indexer = Indexer(zero_start)
        self.weibo_indexer = Indexer(zero_start)
        self.cascades = []
        self.labels = []
        self.content = []
        # self.profiles = []
        self.load()

    def load(self):
        if not self.force_reload and osp.exists(self.save_path):
            with open(self.save_path, 'rb') as f:
                (
                    self.user_indexer,
                    self.weibo_indexer,
                    self.cascades,
                    self.labels,
                    self.content,
                    self._user_profile,
                    version
                ) = pickle.load(f)
            if version != self.version:
                print(f'Version not match, reprocess data.')
                self.process()
        else:
            self.process()

    def save(self):
        with open(self.save_path, 'wb') as f:
            pickle.dump(
                (
                    self.user_indexer,
                    self.weibo_indexer,
                    self.cascades,
                    self.labels,
                    self.content,
                    self._user_profile,
                    self.version
                ), f)

    @property
    def user_profile(self):
        if self._user_profile is None:
            self._user_profile = pd.read_csv(self.user_profile_path, sep='\t', encoding='utf8')
        return self._user_profile

    @property
    def origin_weibo(self):
        if self._origin_weibo is None:
            self._origin_weibo = pd.read_csv(self.origin_weibo_path, sep='\t', encoding='utf8')
        return self._origin_weibo

    @property
    def repost(self):
        if self._repost is None:
            self._repost = pd.read_csv(self.repost_path, sep='\t', encoding='utf8')
        return self._repost

    @property
    def max_sequence_length(self):
        length = max(map(len, self.cascades))
        return length

    def process(self):
        # load user profile
        self.process_dataframe()
        for i, weibo_id in tqdm(enumerate(self.origin_weibo['WeiboId'])):
            # root weibo
            self.cascades.append([])
            self.cascades[-1].append(
                (weibo_id, self.origin_weibo.loc[i, 'UserId'], self.origin_weibo.loc[i, 'WeiboCreateTime'].timestamp()))
            self.content.append(self.origin_weibo.loc[i, 'WeiboText'])
            # no label while test
            if 'ForwordCount' in self.origin_weibo.columns:
                self.labels.append(self.origin_weibo.loc[i, 'ForwordCount'])
            start_time = self.origin_weibo.loc[i, 'WeiboCreateTime']
            end_time = start_time + timedelta(hours=self.observe)
            # multi_task weibo
            for j, row in self.repost.loc[
                self.repost['OriginWeiboId'] == weibo_id].iterrows():
                if row['RepostDate'] > end_time or len(self.cascades[-1]) >= self.max_length:
                    # if len(self.cascades[-1] == 0):
                    break
                self.cascades[-1].append((row['CurrentWeiboId'], row['CurrentUserId'], row['RepostDate'].timestamp()))
                # self.content.append(row['RepostWeiboText'])
        self.save()

    def process_dataframe(self):
        # indexing user id
        self.user_profile['UserId'] = self.user_profile['UserId'].map(self.user_indexer)
        self.origin_weibo['UserId'] = self.origin_weibo['UserId'].map(self.user_indexer)
        self.repost['OriginUserId'] = self.repost['OriginUserId'].map(self.user_indexer)
        self.repost['CurrentUserId'] = self.repost['CurrentUserId'].map(self.user_indexer)
        self.repost['RepostUserId'] = self.repost['RepostUserId'].map(self.user_indexer)
        # indexing weibo id
        self.origin_weibo['WeiboId'] = self.origin_weibo['WeiboId'].map(self.weibo_indexer)
        self.repost['OriginWeiboId'] = self.repost['OriginWeiboId'].map(self.weibo_indexer)
        self.repost['CurrentWeiboId'] = self.repost['CurrentWeiboId'].map(self.weibo_indexer)
        self.repost['RepostWeiboId'] = self.repost['RepostWeiboId'].map(self.weibo_indexer)
        # parse string to datetime
        self.origin_weibo['WeiboCreateTime'] = pd.to_datetime(self.origin_weibo['WeiboCreateTime'])
        self.repost['RepostDate'] = pd.to_datetime(self.repost['RepostDate'])
        self.repost['RepostWeiboText'].fillna('')
        self.repost.sort_values('RepostDate', inplace=True)

    def __getitem__(self, index):
        # (label, [(weibo_id, user_id, timestamp), ...], content)
        label = self.labels[index] if self.mode != 'test' else None
        cas = self.cascades[index]
        r = torch.rand(1).item()
        if self.enhance and r < self.mask_rate:
            cas = cas[:1]
        text = self.content[index]
        return label, cas, text

    def __len__(self):
        return len(self.cascades)

    def subsets(self, rates=(0.7, 0.15, 0.15), shuffle=True):
        subsets = []
        tot = len(self.labels)
        idx = list(range(tot))
        if shuffle:
            np.random.shuffle(idx)
        t = 0
        assert sum(rates) <= 1.0, 'The sum of rates can not be greater than 1.'
        for i, rate in enumerate(rates):
            if i == len(rates) - 1:
                subsets.append(Subset(self, idx[int(t):]))
            else:
                subsets.append(Subset(self, idx[int(t): int(t + rate * tot)]))
            t += rate * tot
        return subsets


if __name__ == '__main__':
    dataset = WeiboRepostDataset(force_reload=False, zero_start=False)
    dataset_test = WeiboRepostDataset(force_reload=False, zero_start=False, mode='test')
    print(dataset[2])
    print(dataset_test[2])
