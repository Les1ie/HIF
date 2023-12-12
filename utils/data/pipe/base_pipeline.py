__all__ = ['Pipeline', 'Filter', ]

from typing import Any

from torch_geometric.transforms import BaseTransform


class Pipeline(BaseTransform):
    added_fields = [

    ]

    def __init__(self, enable=True, context_state=None, **kwargs):
        self.enable = enable
        self.context_state = context_state
        self.kwargs = kwargs

    @property
    def filename_tag(self):
        return ''

    @property
    def is_final_pipeline(self):
        return len(self.added_fields) > 0

    def __call__(self, item=None, state={}, dataset=None, *args, **kwargs) -> Any:
        if self.enable:
            return self.process(item, state, dataset, *args, **kwargs)
        else:
            return item

    def process(self, item, state, dataset, *args, **kwargs) -> Any:
        return item


class Filter(Pipeline):
    @property
    def is_final_pipeline(self):
        return False

    def process(self, item, state=None, dataset=None, *args, **kwargs):
        if self.filter(item=item, state=state, dataset=dataset, *args, **kwargs):
            return None
        else:
            return item

    def filter(self, item, state=None, dataset=None, *args, **kwargs) -> bool:
        return False
