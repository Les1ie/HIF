import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch_geometric.utils import degree

__all__ = ['plot_result', 'collect_result']


def collect_result(epoch_outputs):
    """
    Collects data from epoch_outputs.
    :param epoch_outputs:
    :return:
    """
    data = {'y': [], 'pred': [], '# repost': [], '# follow': []}
    msle = []
    for d in epoch_outputs:
        data['pred'].extend(d['pred'].cpu().relu().numpy().tolist())
        data['y'].extend(d['y'].cpu().numpy().tolist())
        user_batch = d['batch']['user']['batch']
        repost_edge_index = d['batch']['user', 'repost', 'user']['edge_index']
        follow_edge_index = d['batch']['user', 'follow', 'user']['edge_index']
        data['# repost'].extend(degree(user_batch.index_select(0, repost_edge_index[0])).cpu().numpy().tolist())
        data['# follow'].extend(degree(user_batch.index_select(0, follow_edge_index[0])).cpu().numpy().tolist())
        msle.append(d['msle'].item())
    return data, msle


def plot_result(data, title=None, show=True):
    # fig, axs = plt.subplots()
    df = pd.DataFrame(data)
    df['MSLE'] = df.apply(lambda x: (np.log1p(x['pred']) - np.log1p(x['y'])) ** 2, axis=1)
    df.sort_values('y', inplace=True, ignore_index=True)
    ax_edges = df.plot(y=['# repost', '# follow'],
                       kind='line', legend=True, linestyle='--')
    ax = df.plot(x=None, y=['pred', 'y'],
                 xlabel='# Sample', ylabel='Value', logy=True,
                 kind='line', legend=True, ax=ax_edges)
    ax2 = df.plot(x=None, y='MSLE', secondary_y=True, legend=True, kind='area', ax=ax, alpha=0.2)
    ax2.set_ylabel('MSLE')
    ax2.set_ylim(bottom=0)
    ax2.set_zorder(1)
    if title:
        plt.title(title)
    if show:
        plt.show()
    return ax, ax2


if __name__ == '__main__':
    tensor = torch.tensor
    raw_data = [{'loss': tensor(0.0448), 'y': tensor([24, 20, 22, 29, 25, 29, 27, 34]), 'msle': tensor(0.0327),
                 'pred': tensor([31.7421, 20.1269, 31.4158, 28.4335, 19.4150, 30.2272, 28.6970, 31.1371])},
                {'loss': tensor(0.1412), 'y': tensor([33, 27, 24, 25, 20, 23, 31, 48]), 'msle': tensor(0.0990),
                 'pred': tensor([25.0561, 29.7265, 18.9142, 25.7543, 30.6683, 21.4404, 17.1723, 31.5729])},
                {'loss': tensor(0.0513), 'y': tensor([21, 39, 24, 31, 24, 23, 26, 27]), 'msle': tensor(0.0381),
                 'pred': tensor([22.6446, 33.4739, 30.8919, 35.0696, 19.1433, 19.5071, 20.4401, 20.1320])},
                {'loss': tensor(0.0748), 'y': tensor([23, 37, 20, 36, 21, 26, 28, 23]), 'msle': tensor(0.0549),
                 'pred': tensor([23.9814, 31.6207, 18.2327, 26.0138, 28.6321, 17.6820, 22.4831, 28.2342])},
                {'loss': tensor(0.0881), 'y': tensor([21, 29, 24, 20, 22, 20, 24, 20]), 'msle': tensor(0.0663),
                 'pred': tensor([29.6549, 22.6509, 20.0205, 27.1666, 32.3221, 26.4457, 19.6009, 19.4869])},
                {'loss': tensor(0.0660), 'y': tensor([35, 23, 25, 40, 20, 21, 35, 32]), 'msle': tensor(0.0450),
                 'pred': tensor([31.1099, 21.6972, 29.3512, 23.8413, 17.0764, 25.8337, 33.3608, 29.8778])},
                {'loss': tensor(0.0361), 'y': tensor([23, 20, 25, 21, 26, 32, 24, 32]), 'msle': tensor(0.0270),
                 'pred': tensor([19.2114, 29.5057, 27.0896, 22.6482, 29.9083, 30.3573, 23.9284, 28.1560])},
                {'loss': tensor(0.0855), 'y': tensor([31, 24, 24, 28, 24, 27, 40, 33]), 'msle': tensor(0.0608),
                 'pred': tensor([20.6620, 27.1114, 20.5427, 30.3563, 30.7923, 33.0728, 32.5805, 21.9117])},
                {'loss': tensor(0.1923), 'y': tensor([21, 25, 27, 20, 22, 22, 25, 67]), 'msle': tensor(0.1011),
                 'pred': tensor([27.0569, 31.4399, 24.9330, 19.7040, 25.9145, 31.2177, 24.1151, 31.2807])},
                {'loss': tensor(0.1053), 'y': tensor([21, 22, 20, 20, 22, 20, 20, 53]), 'msle': tensor(0.0726),
                 'pred': tensor([23.6680, 23.6652, 25.9478, 30.0096, 19.7619, 29.5423, 27.3805, 37.9159])},
                {'loss': tensor(0.0504), 'y': tensor([22, 24, 32, 22, 20, 20, 22, 24]), 'msle': tensor(0.0365),
                 'pred': tensor([34.3457, 25.4549, 31.4226, 26.5197, 18.8610, 20.8053, 28.6587, 25.2792])}
                ]
    data = {'y': [], 'pred': []}
    for d in raw_data:
        data['pred'].extend(d['pred'].numpy().tolist())
        data['y'].extend(d['y'].numpy().tolist())
    plot_result(data)
