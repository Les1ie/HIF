import torch

__all__ = ['gen_idx', 'exp_dilation', 'fib_dilation', 'sin_dilation', ]


def gen_idx(ks):
    idx = torch.arange(1, 1 + ks)
    return idx


def exp_dilation(kernel_size, alpha=0.3):
    if kernel_size == 0:
        return []
    idx = gen_idx(kernel_size)
    di = torch.exp(alpha * idx).to(torch.int)
    return di


def fib_dilation(kernel_size):
    if kernel_size == 0:
        return []
    fib = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
    if kernel_size > len(fib):
        for i in range(kernel_size - len(fib)):
            fib.append(fib[-1] + fib[-2])
    return torch.tensor(fib[:kernel_size])


def sin_dilation(kernel_size, T=2.1, alpha=-1, w=1.):
    if kernel_size == 0:
        return []
    idx = gen_idx(kernel_size)
    di = 1 + w + w * torch.cos(idx * torch.pi / T + (alpha - T) * torch.pi / T)
    di = di.to(dtype=torch.int)
    return di


if __name__ == '__main__':
    ks = 15
    # for a in torch.arange(0.3, 0.7, 0.05):
    #     print('alpha={:.2}:'.format(a), exp_dilation(ks, a))

    # print('fib({}): '.format(ks), fib_dilation(ks).tolist())
    # print('exp({}): '.format(ks), exp_dilation(ks).tolist())
    # print('sin({}): '.format(ks), sin_dilation(ks).tolist())
    for i in [1+j/100 for j in range(0, 201, 10)]:
        print('sin({:2.3}):\t'.format(i), sin_dilation(ks, i, ).tolist())
