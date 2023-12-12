from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import MeanSquaredLogError

from models import Predictor
from nn import TemporalConvNet
from nn.tcn.nltcn import NonLinearTemporalConvNet
from utils.data import sequence_padding_collate
from utils.data.datamodule import DataModule


class TCNPredictor(Predictor):
    name = 'TCN'

    # name = 'TimeConvolutionalNetwork'
    def __init__(self, in_feats=32, hid_feats=64, out_feats=1, num_users=None,
                 kernel_size=3, num_layers=3,
                 learning_rate=5e-3, weight_decay=5e-4, patience=10, dropout=0.1, data_name=None):
        super().__init__(in_feats, hid_feats, out_feats, learning_rate, weight_decay, dropout, patience,data_name)
        self.num_user = num_users
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.user_embeddings = nn.Embedding(num_users + 1, in_feats)
        self.tcn = TemporalConvNet(in_feats, [in_feats] * num_layers, kernel_size, dropout=dropout)
        self.output_layers = nn.Sequential(
            nn.Linear(in_feats, hid_feats), nn.BatchNorm1d(hid_feats), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hid_feats, hid_feats), nn.BatchNorm1d(hid_feats), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hid_feats, out_feats), nn.CELU(0.5)
        )
        self.loss = MeanSquaredLogError()
        self.save_hyperparameters(ignore=['total_nodes'])

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = super(TCNPredictor, TCNPredictor).add_model_specific_args(parent_parser)
        arg_parser = parent_parser.add_argument_group('TCNPredictor')
        arg_parser.add_argument("--num_layers", type=int, default=3)
        arg_parser.add_argument("--kernel_size", type=int, default=3)
        return parent_parser

    def forward(self, seq) -> Any:
        uid = seq[:, :, 1]
        x = self.user_embeddings(uid)
        x = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        x = torch.sum(x, 1)
        pop = self.output_layers(x)
        return pop


class NlTCNPredictor(TCNPredictor):
    name = 'NL-TCN'

    # name = 'NonLinearTimeConvolutionalNetwork'
    def __init__(self, in_feats=32, hid_feats=32, out_feats=1, num_users=None,
                 dilation_func='sin', kernel_size=3, num_layers=3,
                 learning_rate=5e-3, weight_decay=5e-4,
                 patience=10, dropout=0.1, data_name=None):
        super().__init__(in_feats, hid_feats, out_feats, num_users,
                         kernel_size, num_layers, learning_rate,
                         weight_decay, patience, dropout, data_name)
        self.dilation_func = dilation_func
        self.tcn = NonLinearTemporalConvNet(in_feats, in_feats,
                                            num_layers=num_layers, kernel_size=kernel_size,
                                            dilation_func=dilation_func)
        self.save_hyperparameters('dilation_func')

    def forward(self, seq) -> Any:
        uid = seq[:, :, 1]
        x = self.user_embeddings(uid)
        x = self.tcn(x)
        x = torch.sum(x, 1)
        pop = self.output_layers(x)
        return pop


if __name__ == '__main__':
    for kernel_size in [3, 4, 5]:
        for num_layers in [3, 4]:
            # for dilation in ['sin', 'fib', 'exp']:
                pl.seed_everything(0)
                data_name = 'multi_task'
                data_module = DataModule(batch_size=32,
                                     num_workers=0,
                                     collate_fn=sequence_padding_collate,
                                     root='/root/hif/data',
                                     sample=0.05, name=data_name, persist='memory',
                                     force_reload=False,
                                     )
            # model = NlTCNPredictor(total_nodes=data_module.dataset.total_nodes,
            #                        kernel_size=kernel_size,
                #                        num_layers=num_layers,
                #                        dilation_func=dilation)
                model = TCNPredictor(num_users=data_module.dataset.num_users,
                                     kernel_size=kernel_size,
                                     num_layers=num_layers)
                trainer = model.configure_trainer()
                trainer.fit(model, data_module)
                trainer.test(model, data_module)

    # pl.seed_everything(0)
    # parser = DataModule.add_argparse_args(ArgumentParser())
    # args, unknown_args = parser.parse_known_args()
    # data_module = DataModule.from_argparse_args(args,
    #                                             collate_fn=sequence_padding_collate,
    #                                             root='data\\multi_source',
    #                                             sample=0.05, name='multi_task', persist='memory',
    #                                             force_reload=False,
    #                                             )
    # parser = TCNPredictor.configure_argument_parser()
    # args, unknown_args = parser.parse_known_args(unknown_args)
    # model = NlTCNPredictor(**vars(args), total_nodes=data_module.dataset.total_nodes, dilation_func='fib')
    # # model = TCNPredictor(**vars(args), total_nodes=data_module.dataset.total_nodes)
    # trainer, unknown_args = model.configure_trainer(unknown_args)
    #
    # trainer.fit(model, data_module)
    # trainer.test(model, data_module)
