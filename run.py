import traceback
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import pytorch_lightning as pl
import torch_geometric
from torch.utils.data import DataLoader

from models import *
from utils.data import DataModule, RandomEmbedCollater
from utils.data.datasets import CSVHeteroGraphDataset
from os import path as osp


def load_hps(hp_path):
    import yaml
    hps = yaml.load(open(hp_path, 'r'), yaml.Loader)
    return hps

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    p = parser.add_argument_group('Running')
    p.add_argument('--model', type=str, default='TSA', choices=['TSA', 'TA'], help='The model name.')
    p.add_argument('--test_ckpt', type=str, default=None, help='Path of checkpoint to test without training.')
    p.add_argument('--seed', type=int, default=0, help='Random seed.')

    predictor_cls = TimestampAttendPredictor
    parser = predictor_cls.configure_argument_parser(parser)
    parser = DataModule.add_argparse_args(parser)
    args = parser.parse_args()
    args.predictor_cls = predictor_cls
    if args.test_ckpt == 'None':
        args.test_ckpt = None
    return args


def run(args):
    dict_args = vars(args)
    dict_args.update(dict_args.get('all_args', {}))
    dict_args['activate_cls'] = nn.LeakyReLU
    # dict_args['max_length'] = 500
    ckpt_path = args.test_ckpt
    load_ckpt = ckpt_path is not None
    pl.seed_everything(dict_args.get('seed', 0))
    try:
        if load_ckpt:
            model: Predictor = dict_args['predictor_cls'].load_from_checkpoint(ckpt_path)

            if model.all_args['learnable_embedding']:
                collate_fn = None
                dataloader_cls = torch_geometric.loader.DataLoader
            else:
                collate_fn = RandomEmbedCollater(dim=model.in_feats)
                dataloader_cls = torch.utils.data.DataLoader

            '''
            加载数据集并process
            '''
            data_args = model.data_args
            print("Data Args:", data_args)
            data_module = DataModule(**data_args, dataloader_cls=dataloader_cls,
                                     collate_fn=collate_fn,
                                     root=data_args.get('root', '/root/hif/data'),
                                     dataset_cls=CSVHeteroGraphDataset)
            data_module.prepare_data()  # reset random seed
            print("Data Module Created")
            pl.seed_everything()
        else:

            if dict_args['learnable_embedding']:
                collate_fn = None
                dataloader_cls = torch_geometric.loader.DataLoader
            else:
                collate_fn = RandomEmbedCollater(dim=dict_args['in_feats'])
                dataloader_cls = torch.utils.data.DataLoader

            data_args = dict_args
            data_module = DataModule(**data_args, dataloader_cls=dataloader_cls,
                                     collate_fn=collate_fn,
                                     dataset_cls=CSVHeteroGraphDataset)
            data_module.prepare_data()  # reset random seed
            pl.seed_everything()
            model: Predictor = dict_args['predictor_cls'](**dict_args,
                                                          all_args=dict_args,
                                                          data_args=data_module.data_args(),
                                                          total_nodes=data_module.dataset.total_nodes,
                                                          # total_nodes=data_module.dataset_cls.total_nodes,
                                                          meta_edges=data_module.dataset.meta_edges)
                                                          # meta_edges=data_module.dataset_cls.meta_edges)

        pl.seed_everything()
        trainer = model.configure_trainer(args, no_logger=ckpt_path is not None)
        pl.seed_everything()
        if not load_ckpt:
            trainer.tune(model, datamodule=data_module)
            trainer.fit(model, datamodule=data_module)
            pl.seed_everything()
        rst = trainer.test(model, datamodule=data_module)
        return rst
    except Exception as e:
        traceback.print_exc()
        print(dict_args)




if __name__ == '__main__':
    result = run(parse_args())[0]
    print(result['hp_metric'])
