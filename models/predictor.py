from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from typing import Any, Optional, Union
from os import path as osp

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch import nn
from torchmetrics import MeanSquaredLogError
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn import functional as F
__all__ = ['Predictor']


class Predictor(pl.LightningModule):
    name = None

    def __init__(self,
                 in_feats=16, out_feats=1,
                 learning_rate=5e-3, weight_decay=5e-4, dropout=0.1, l1_weight=5e-4,
                 patience=10, loss_func=MeanSquaredLogError, activate_cls=nn.LeakyReLU,
                 data_args=None, all_args=None,
                 log_weight=False, log_grad=False,
                 **kwargs
                 ):
        self.in_feats = in_feats
        # self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.l1_weight = l1_weight
        self.patience = patience
        self.data_args = data_args
        self.log_weight = log_weight
        self.log_grad = log_grad
        self.all_args = all_args  # save all args for initialize modules
        super(Predictor, self).__init__()
        self.dropout_func = nn.Dropout(dropout)
        self.loss_func = loss_func() if isinstance(loss_func, type) else loss_func
        self.activate_cls = activate_cls if isinstance(activate_cls, type) else nn.LeakyReLU
        self.activate_func = self.activate_cls()
        self.l1_loss = nn.MSELoss()
        self.save_hyperparameters('in_feats', 'out_feats',
                                  'learning_rate', 'weight_decay', 'dropout',
                                  'patience', 'data_args', 'loss_func', 'activate_cls', 'all_args', )

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        arg_parser = parent_parser.add_argument_group("Predictor")
        arg_parser.add_argument("--in_feats", type=int, default=16, help='Dimension of inputs.')
        arg_parser.add_argument("--out_feats", type=int, default=1, help='Dimension of output layers.')
        arg_parser.add_argument("--learning_rate", type=float, default=5e-3,
                                help='Learning rate to optimize parameters.')
        arg_parser.add_argument("--weight_decay", type=float, default=5e-4, help='Weight decay (L2 loss).')
        arg_parser.add_argument("--l1_weight", type=float, default=5e-4, help='L1 loss.')
        arg_parser.add_argument("--dropout", type=float, default=0.1, help='Dropout.')
        arg_parser.add_argument("--log_weight", action="store_true", default=False,
                                help='Log all weights on tensorboard.')
        arg_parser.add_argument("--log_grad", action="store_true", default=False,
                                help='Log all grad on tensorboard.')
        arg_parser.add_argument("--patience", type=int, default=10, help='Patience of early stopping.')
        return parent_parser

    def forward(self, *args, **kwargs) -> Any:
        return

    def metrics(self, stage, *args, **kwargs):
        pred = kwargs.get('pred')
        labels = kwargs.get('y')
        msle = self.loss_func(F.celu(pred, 0.5), labels)
        l1 = self.l1_weight * self.l1_loss(pred, labels)
        self.log(f'l1_loss/{stage}', l1, on_step=True, on_epoch=False, prog_bar=False)
        return {
            'loss': msle + l1,
            'msle': msle,
            'l1_loss': l1,
        }

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        labels, seq = args[0]
        pred = self(seq)
        metrics = self.metrics(stage='train', pred=pred, y=labels)
        loss = metrics['loss']
        self.log('loss/train', loss)
        return loss

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        labels, seq = args[0]
        pred = self(seq)
        metrics = self.metrics(stage='val', pred=pred, y=labels)
        loss = metrics['loss']
        self.log('loss/val', loss)
        return loss

    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        labels, seq = args[0]
        pred = self(seq)
        metrics = self.metrics(stage='test', pred=pred, y=labels)
        loss = metrics['loss']
        self.log('loss/test', loss)
        self.log('hp_metric', metrics['msle'], on_step=False, on_epoch=True)
        return {'loss':loss,
                'labels':labels,
                'pred':pred}

    def configure_callbacks(self):
        early_stop = EarlyStopping(monitor="loss/val", mode="min", patience=self.patience)
        checkpoint = ModelCheckpoint(monitor="loss/val", save_top_k=1, save_last=True, mode='min',
                                     filename='epoch={epoch}-val_loss={loss/val:.2f}',
                                     auto_insert_metric_name=False,
                                     )
        lr_monitor = LearningRateMonitor()
        return [early_stop, checkpoint, lr_monitor]

    def configure_optimizers(self):
        adam = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        cosine_annealing = CosineAnnealingWarmRestarts(adam, T_0=5, T_mult=3)
        return [adam], [cosine_annealing]

    def configure_trainer(self, arg_list: Union[ArgumentParser, Namespace] = None, no_logger=False) -> Trainer:
        if no_logger:
            tb_logger = None
        else:
            data_args = self.data_args
            sub_dir_name = data_args['name'] + str(data_args['observation']) + 'h'
            tb_logger = TensorBoardLogger(osp.join('lightning_logs', sub_dir_name)
                                          , name=self.name)
        if arg_list:
            trainer = Trainer.from_argparse_args(arg_list, logger=tb_logger)
            return trainer
        else:
            trainer = Trainer(logger=tb_logger)
            return trainer

    @classmethod
    def configure_argument_parser(cls, argument_parser=None) -> ArgumentParser:
        if argument_parser is None:
            argument_parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
        argument_parser = cls.add_model_specific_args(argument_parser)
        argument_parser = Trainer.add_argparse_args(argument_parser)
        return argument_parser

    def log_weights(self, grad=True):
        # log grad and weights
        for name, param in self.named_parameters():
            try:
                if self.log_weight:
                    self.logger.experiment.add_histogram(f'weights/{name}',
                                                         # param.cpu(),
                                                         param.clone().cpu().data.numpy(),
                                                         global_step=self.trainer.global_step)
                if grad and param.grad is not None and self.log_grad:
                    self.logger.experiment.add_histogram(f'grad/{name}',
                                                         # param.grad.cpu(),
                                                         param.grad.clone().cpu().data.numpy(),
                                                         global_step=self.trainer.global_step)
            except:
                pass
