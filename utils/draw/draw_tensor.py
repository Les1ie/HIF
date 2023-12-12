from matplotlib import pyplot as plt
import torch
from matplotlib import cm

__all__ = ['plot_2d_tensor']


def plot_2d_tensor(tensor: torch.Tensor, cmap=cm.binary,
                   show=False,
                   **kwargs):
    arr = tensor.numpy()
    fig, ax = plt.subplots(**kwargs)
    ax.matshow(arr, cmap=cmap)
    if show:
        plt.show()
    return fig, ax


if __name__ == '__main__':
    t = torch.normal(0, 1, [32, 64])
    fig, ax = plot_2d_tensor(t, figsize=(4, 3))
    plt.show()
    print(fig, ax)
