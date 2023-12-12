import argparse
import shutil
import os
from os import path as osp

tot = 0


def clean(path, depth, max_depth):
    global tot
    null_ckpt = False
    for f in os.listdir(path):
        p = osp.join(path, f)
        if osp.isfile(p):
            continue
        if depth > max_depth:
            continue
        to_clean, reason = clean(p, depth + 1, max_depth)
        if to_clean:
            shutil.rmtree(p)
            print('remove %r, %s' % (p, reason))
            tot += 1
            if f == 'checkpoints':
                null_ckpt = True

    l = list(os.listdir(path))
    if len(l) == 0:
        return True, 'empty dir'
    if osp.split(path)[-1].startswith('version_') and 'checkpoints' not in l:
        return True, 'no checkpoints'
    if null_ckpt:
        return True, 'no checkpoints'

    return False, "it's ok"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='lightning_logs')
    parser.add_argument('--depth', type=int, default=4)
    args = parser.parse_args()
    dict_args = vars(args)
    if osp.exists(args.dir):
        clean(args.dir, 0, args.depth)
        print(f'{tot} dirs has been removed.')
