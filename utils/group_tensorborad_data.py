from argparse import ArgumentParser

import pandas as pd
import os
from os import path as osp


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', type=str)
    parser.add_argument('-g', '--group', type=str, action='extend', nargs='+',
                        default=['data_args/name', 'data_args/observation'])
    parser.add_argument('-c', '--compact', action='store_true', default=False, help='是否仅保留用于group的参数列')
    parser.add_argument('-m', '--metric', type=str, default='hp_metric', help='用于group min操作的指标')

    args = parser.parse_args()
    args.group = list(set(args.group))
    df = pd.read_csv(args.file)
    df = df[df[args.metric] > 0]
    print(df.head())
    result = df.groupby(args.group, as_index=False).min(args.metric)
    if args.compact:
        cols = args.group + [args.metric]
        result = result.filter(items=cols)
        result.rename(lambda x: x.split('/')[-1], axis=1, inplace=True)
    print(result)

