from typing import Any

from torch import nn

__all__ = ['ConvMixin']


class ConvMixin(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.convs = nn.ModuleList(self.get_convs())
        # self.callbacks_on_conv

    def get_convs(self, *args, **kwargs) -> Any:
        return []

    def on_conv(self, *args, **kwargs) -> Any:
        pass

    def conv(self, *args, **kwargs) -> Any:
        x = kwargs['x']
        batch = kwargs['batch']
        edge_index_dict = kwargs['edge_index_dict']
        self.on_conv(x=x, batch=batch)
        for step, conv in enumerate(self.convs):
            self.on_conv_step(x=x, edge_index_dict=edge_index_dict,
                              batch=batch, conv=conv, conv_step=step)
            x, edge_index_dict = self.conv_step(x=x, edge_index_dict=edge_index_dict,
                                                batch=batch, conv=conv, conv_step=step)
            self.on_conv_step_end(x=x, edge_index_dict=edge_index_dict,
                                  batch=batch, conv=conv, conv_step=step)
        self.on_conv_end(x=x, batch=batch, edge_index_dict=edge_index_dict, )
        result = {'x': x, 'batch': batch, 'edge_index_dict': edge_index_dict}
        return result

    def conv_step(self, *args, **kwargs) -> Any:
        pass

    def on_conv_step(self, *args, **kwargs) -> Any:
        pass

    def on_conv_step_end(self, *args, **kwargs) -> Any:
        pass

    def on_conv_end(self, *args, **kwargs) -> Any:
        pass
