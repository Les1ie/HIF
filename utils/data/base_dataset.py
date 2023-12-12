import os
import pickle
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from os import path as osp
from typing import Any, Optional

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from tqdm.contrib.concurrent import process_map

from utils import AdditiveDict
from utils.data import pipe

__all__ = ['PipelineDataset']


class PipelineDataset(Dataset):
    _saved_fields = set(['context_state'])
    auto_load = True
    save_name_split = '-'

    def __init__(self, root, name=None, pipelines=None, force_reload=False,
                 transform: Optional[list] = None, num_workers=0, pool='thread',
                 raw_items=[],
                 persist='memory', **kwargs):
        self.kwargs = kwargs
        self.transform = transform
        self.num_workers = num_workers
        self.pool = pool
        self.root = root
        self._name = name
        self.force_reload = force_reload
        self.context_state = AdditiveDict(raw_items=[], raw_item_idx=[], on_error='replace')
        kwargs['context_state'] = self.context_state
        if pipelines is None:
            pipelines = []
        else:
            pipelines = [p(**kwargs) for p in pipelines]

        self.pipelines = pipelines
        self.persist = persist
        self._raw_items = raw_items
        self._processed_items = []
        # todo: persist意义不明，与self.save()冲突，相关参数暂不使用
        expt_persist = ['memory', 'store']
        persist = 'memory'
        if persist not in expt_persist:
            raise Exception(f'Excepted "persist" in {expt_persist}, but {persist} received.')
        if persist == 'memory':
            self.pipelines.append(
                pipe.InmemoryPersistPipeline(self._processed_items, **kwargs))
        else:
            save_dir = osp.join(self.save_dir, self.save_name.replace('.pkl', '.items'))
            self.pipelines.append(
                pipe.StorePersistPipeline(target_dir=save_dir, **kwargs))
        self.check_pipelines()
        self.processed_context_state = deepcopy(self.context_state)
        self.__has_processed = False
        if self.auto_load:
            self._load()

    def check_pipelines(self):
        final_tag = False
        for p in self.pipelines:
            if p.is_final_pipeline:
                final_tag = True
            if final_tag and isinstance(p, pipe.Filter):
                raise Exception(f'Filters can not be behind of final pipelines.')

    def add_pipeline(self, pipeline):
        self.pipelines.insert(-1, pipeline)
        self.check_pipelines()

    def read_raw_items(self):
        pass

    @property
    def raw_items(self) -> Any:
        if len(self._raw_items) == 0:
            self.read_raw_items()
        return self._raw_items

    def has_processed(self) -> bool:
        if not osp.exists(self.save_path):
            return False
        if self.persist == 'store':
            cnt = len(list(filter(lambda x: x.endswith('.pkl'), os.listdir(self.persist_pipeline.target_dir))))
            # if cnt != self.number_of_processed_items:
            #     return False
        return True

    def process(self):
        print(f'Dataset save path: {self.save_path}')
        if self.num_workers:
            pool = __import__(f'tqdm.contrib.concurrent.{self.pool}_pool')
            from tqdm.contrib.concurrent import thread_map as pool
            def fn(x):
                self._process(raw_item=x[1], raw_item_idx=x[0])

            pool(fn, list(enumerate(self.raw_items)),
                 desc=f'Processing {self.name}', total=len(self.raw_items))
        else:
            from tqdm.auto import tqdm
            with tqdm(desc=f'Processing {self.name}', total=len(self.raw_items)) as bar:
                for raw_item_idx, raw_item in enumerate(self.raw_items):
                    self._process(raw_item=raw_item, raw_item_idx=raw_item_idx)
                    bar.update(1)
            self.context_state = self.processed_context_state
            # self._sync_fields()

    def _process(self, raw_item_idx, raw_item):
        item = raw_item
        # local_state = deepcopy(self.context_state)
        local_state = deepcopy(self.processed_context_state)
        local_state['raw_items'].append(raw_item)
        local_state['raw_item_idx'].append(raw_item_idx)
        for pipe_idx, pipeline in enumerate(self.pipelines):
            item = pipeline(item=item, state=local_state, dataset=self,
                            raw_item_idx=raw_item_idx, raw_item=raw_item)
            if item is None:
                break
            if isinstance(pipeline, pipe.Filter) and item is None:
                break
        if item:
        # if item and self.context_state is not local_state:
        #     self.context_state = local_state
        #     self.processed_context_state += local_state     # todo: 留意 Indexer 不支持 add 操作导致的 BUG
            self.processed_context_state = local_state
        return item

    def _load(self):
        if self.force_reload or not self.has_processed():
            if not self.__has_processed:
                self.process()
                self.__has_processed = True
                self.save()
        else:
            self.load()

    def load(self):
        with open(self.save_path, 'rb') as f:
            values = pickle.load(f)
            f.close()
        for k, v in values.items():
            self.__setattr__(k, v)

    def _sync_fields(self):
        for field, pipeline in self.added_fields.items():
            self.__setattr__(field, getattr(pipeline, field))

    @property
    def save_dir(self):
        d = osp.join(self.root, 'processed')
        if not osp.exists(d):
            os.mkdir(d)

        return d

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def name(self):
        return self._name

    @property
    def save_name_tags(self):
        tags = [self.name]
        tags += [p.filename_tag for p in self.pipelines if p.enable and len(p.filename_tag)]
        return tags

    @property
    def save_name_suffix(self):
        return '.pkl'

    @property
    def save_name(self):
        name = self.save_name_split.join(self.save_name_tags) + self.save_name_suffix
        return name

    @property
    def save_path(self):
        p = osp.join(self.save_dir, self.save_name)
        return p

    @property
    def saved_fields(self) -> set:
        fields = self._saved_fields
        for p in self.pipelines:
            fields.update(p.added_fields)
        fields.update(['_processed_items', '_raw_items'])
        return fields

    def save(self):
        d = {
            k: getattr(self, k) for k in self.saved_fields
        }
        with open(self.save_path, 'wb') as f:
            pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
            f.close()

    @property
    def processed_items(self):
        return self._processed_items

    def __getitem__(self, index) -> T_co:

        if self.persist == 'memory':
            item = self.processed_items[index]
        else:
            if index >= self.number_of_processed_items:
                raise IndexError
            p = self.persist_pipeline.persist_path(index)
            with open(p, 'rb') as f:
                item = pickle.load(f)
        return self.transform_item(item)

    def transform_item(self, item):
        it = item
        if self.transform is not None:
            for t in self.transform:
                it = t(it)
        return it

    @property
    def persist_pipeline(self):
        return self.pipelines[-1]

    def __len__(self) -> int:
        if self.persist == 'memory':
            return len(self._processed_items)
        else:
            return self.number_of_processed_items

    def __repr__(self):
        name = type(self).__name__
        return f'{name}({len(self)}, name={self.name}, root={self.root}, persist={self.persist})'


class ParallelPipelineDataset(Dataset):
    def __init__(self):
        super(ParallelPipelineDataset, self).__init__()
