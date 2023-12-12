from collections import Iterable


class Indexer:
    def __init__(self, zero_start=True):
        '''
        :param zero_start: Whether indexes begin from zero.
        '''
        self.item2id = dict()
        self.id2item = []
        if not zero_start:
            self.id2item.append(None)

    def add_item(self, item):
        '''
        Index an item.
        :param item: item to index.
        :return: index of input item.
        '''
        if not self.item_exists(item):
            self.item2id[item] = len(self.id2item)
            self.id2item.append(item)
        return self.item2id[item]

    def item_exists(self, item):
        '''
        Whether an item has been indexed.
        :param item:
        :return:
        '''
        return item in self.item2id.keys()

    def __contains__(self, item):
        return self.item_exists(item)

    def index_items(self, items: Iterable):
        '''
        Index multiple items.
        :param items: items to index.
        :return: indexes of items.
        '''
        idx = []
        for it in items:
            idx.append(self(it))
        return idx

    def get_item(self, idx):
        '''
        Get indexed item by id.
        :param idx:
        :return:
        '''
        return self.id2item[idx]

    def get_id(self, item):
        '''
        Get item's index.
        :param item:
        :return: the index of item, return None if item is not indexed.
        '''
        if item in self:
            return self.item2id[item]
        else:
            return None

    def reset(self):
        self.item2id = dict()
        self.id2item = []

    def __call__(self, item):
        self.add_item(item)
        item_id = self.get_id(item)
        return item_id

    def __getitem__(self, index):
        return self.id2item[index]

    def __len__(self):
        '''
        Total number of indexed items.
        :return:
        '''
        return len(self.id2item)


if __name__ == '__main__':
    # test
    indexer = Indexer()
    l = ['A', 'B', 'C', 'D', 'E']
    l2 = ['A', 'B', 'asd', 'qwe', 'ASD']
    indexer.index_items(l)
    print(indexer.id2item)
    print(indexer.item2id)
    indexer.index_items(l2)
    print(indexer.id2item)
    print(indexer.item2id)
