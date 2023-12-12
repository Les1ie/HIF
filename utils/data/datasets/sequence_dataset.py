import os
import random
from os import path as osp

from utils.data.base_dataset import PipelineDataset
from utils import Indexer
from utils.data import pipe

__all__ = ['CSVSequenceDataset']


class CSVSequenceDataset(PipelineDataset):
    _saved_fields = {'indexer'}

    def __init__(self, root, name=None, force_reload=False, persist='memory',
                 observation=2, relative_timestamp=False, sample=0.1, **kwargs
                 ):
        self.observation = observation
        self.relative_timestamp = relative_timestamp
        self.sample = sample
        self.indexer = Indexer(zero_start=False)
        pipes = [
            pipe.CSV2DataframePipeline(),
            pipe.ExtractLabelPipeline(),
            pipe.ObservationPipeline(observation),
            pipe.IndexDataframePipeline(self.indexer),
            pipe.Dataframe2SequencePipeline(relative_timestamp),
        ]
        super().__init__(root, name, force_reload=force_reload, pipelines=pipes, persist=persist)

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw', f'{self.name}_cascades')

    @property
    def save_name(self):
        values = {
            'name': self.name,
            'persist': self.persist,
            'observation': self.observation,
            'sample': self.sample,
        }
        name = '{name}-{observation}-{persist}{sample}.seq.pkl'.format(**values)
        return name

    @property
    def num_users(self):
        num = len(self.indexer)
        return num

    def read_raw_items(self):
        file_list = list(filter(lambda x: x.endswith('csv'), os.listdir(self.raw_dir)))
        tot = len(file_list)
        cnt = int(tot * self.sample)
        file_list = random.sample(file_list, cnt)
        self._raw_items = list(file_list)
        super().read_raw_items()

    def __getitem__(self, index):
        item = super(CSVSequenceDataset, self).__getitem__(index=index)
        label = self.labels[index]
        return item, label


if __name__ == '__main__':
    dataset = CSVSequenceDataset('/root/hif/data', name='multi_task', persist='memory',
                                 force_reload=True,
                                 sample=0.05
                                 )
    print(dataset)
    print(len(dataset))
    for i, c in enumerate(dataset):
        print(i, c)
