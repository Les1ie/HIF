import os
import sys
from collections import Counter
from os import path as osp
from typing import Optional, Any, Sequence
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, BasePredictionWriter
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics import MeanSquaredLogError
from nn import WeightedPrecision, TemporalConvNet
from nn.positional_encoding import TimeAwarePositionalEncoding
from utils.data.datasets.weibo_repost_dataset import WeiboRepostDataset

sys.path.append('.')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

__all__ = ['Transpose', 'PopularityClassifier']


class Transpose(nn.Module):
    def __init__(self, dims=(0, 1)):
        super(Transpose, self).__init__()
        self.dims = dims

    def forward(self, x):
        return torch.transpose(x, *self.dims)


class PopularityClassifier(pl.LightningModule):

    def __init__(self, in_feats=5, hid_feats=16, out_feats=5,
                 num_encoder_layers=4, num_attention_heads=8,
                 dropout=0.1, learning_rate=5e-3, weight_decay=1e-4, batch_size=16,
                 pooling='sum', label_smoothing=0.1,
                 embedding=None,
                 tcn_kernel_size=3) -> None:
        super().__init__()
        # attributes
        self._ = None
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.dropout = dropout
        self.num_encoder_layers = num_encoder_layers
        self.num_attention_heads = num_attention_heads
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.pooling = pooling
        self.label_smoothing = label_smoothing
        self.tcn_kernel_size = tcn_kernel_size

        self._user_embeddings = embedding
        # metrics and loss function
        self.loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,  # label smoothing
        )
        self.msle = MeanSquaredLogError()
        '''
        权重交叉熵用于定向的提高召回率，适用于专门的业务需求
        不适用于样本不均匀导致的性能问题，且和重采样搭配使用时容易导致梯度过高
        对于
        '''
        # self.loss = nn.CrossEntropyLoss(
        #   weight=torch.tensor([1., 10., 50., 100., 300.]),    # use weighted cross entropy
        #   label_smoothing=label_smoothing,    # label smoothing
        #                                 )
        self.weighted_precision = WeightedPrecision()

        # layers
        self.input_transform = nn.Sequential(
            # nn.BatchNorm1d(in_feats),
            nn.Linear(in_feats, hid_feats),
            Transpose(dims=(1, 2)),
            nn.BatchNorm1d(hid_feats),
            Transpose(dims=(1, 2)),
            # nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hid_feats, hid_feats),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        self.output_layers_classification = nn.Sequential(nn.Linear(hid_feats, hid_feats),
                                                          nn.LayerNorm(hid_feats),
                                                          nn.Dropout(dropout),
                                                          nn.ReLU(),
                                                          nn.Linear(hid_feats, out_feats),
                                                          # nn.LogSoftmax(1),
                                                          )
        self.output_layers_regression = nn.Sequential(nn.Linear(hid_feats, hid_feats),
                                                      nn.LayerNorm(hid_feats),
                                                      nn.Dropout(dropout),
                                                      nn.ReLU(),
                                                      nn.Linear(hid_feats, 1),
                                                      )
        self.positional_encoding = TimeAwarePositionalEncoding(hid_feats, )
        encoder_layer = nn.TransformerEncoderLayer(d_model=hid_feats, nhead=num_attention_heads, batch_first=True,
                                                   dropout=dropout)
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers,
                                                      norm=nn.LayerNorm(hid_feats))
        self.tcn = TemporalConvNet(hid_feats, [hid_feats, hid_feats, hid_feats, ],
                                   dropout=dropout, kernel_size=tcn_kernel_size)

        self.layer_norm = nn.LayerNorm(hid_feats)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)
        self.norm_act_drop = nn.Sequential(self.layer_norm, self.relu, self.drop)

        # self.tcn_attn = nn.MultiheadAttention(hid_feats, num_heads=num_attention_heads,
        #                                       dropout=dropout, batch_first=True)
        # self.text_encoder = BertEncoder()
        # bert_output_size = 768
        # self.text_mlp = nn.Linear(bert_output_size, hid_feats)
        # self.text_atten = nn.MultiheadAttention(hid_feats, num_heads=self.num_attention_heads, dropout=dropout,
        #                                         # kdim=bert_output_size, vdim=bert_output_size,
        #                                         batch_first=True)

        self.save_hyperparameters()

    @property
    def user_embeddings(self):
        return self._user_embeddings

    @user_embeddings.setter
    def user_embeddings(self, user_profile: pd.DataFrame):
        user_profile['Verified'] = user_profile['Verified'].map(lambda x: int(x))
        user_profile['Gender'] = user_profile['Gender'].map(lambda x: 1 if x == 'f' else 0)
        self._user_embeddings = torch.tensor(user_profile.iloc[:, 1:5].to_numpy(), dtype=torch.float)
        # padding user's embedding
        zeros = torch.zeros((1, self._user_embeddings.shape[1]), dtype=self._user_embeddings.dtype)
        self._user_embeddings = torch.cat([zeros, self._user_embeddings], 0).to(device=self.device)

    def log_metric(self, metric_name, val, ignores={}, **kwargs):
        modes = {'evaluating': self.trainer.evaluating,
                 'training': self.trainer.training,
                 'predicting': self.trainer.predicting,
                 'testng': self.trainer.testing
                 }
        default_ignores = {'evaluating': False,
                           'training': False,
                           'predicting': True,
                           'testng': True
                           }
        default_ignores.update(ignores)
        mode_name = {'evaluating': 'Valid',
                     'training': 'Train',
                     'predicting': 'Predict',
                     'testng': 'Test'}

        for k, v in modes.items():
            if default_ignores[k] and not v:
                continue

            log_name = f'{mode_name[k]} {metric_name}'
            self.log(log_name, val, **kwargs)

    def forward(self, sequence, seq_len, text_seq) -> Any:
        # extracting user index to get user embeddings
        root_idx = sequence[:, 0, 1]
        user_idx = sequence[:, :, 1].to(dtype=torch.long)
        seq_x = self.user_embeddings[user_idx]
        # append timestamps
        seq_time = sequence[:, :, 2].to(dtype=torch.float)
        seq_time = torch.unsqueeze(seq_time, 2)
        seq_x = torch.cat([seq_x, seq_time], 2)

        # forward computing
        seq_x = self.input_transform(seq_x)
        # time positional encoding
        seq_x_pe = self.positional_encoding(seq_x, seq_time)
        seq_x_pe = self.relu(self.layer_norm(seq_x_pe))

        # tcn
        time_conv = self.tcn(torch.transpose(seq_x_pe, 1, 2))
        time_conv_transpose = self.norm_act_drop(seq_x_pe + torch.transpose(time_conv, 1, 2))
        # seq_x = time_attn

        # transformer encoding
        seq_encode = time_conv_transpose + self.sequence_encoder(time_conv_transpose)
        seq_encode = self.norm_act_drop(seq_encode)
        # time_attn, time_attn_weights = self.tcn_attn(seq_x, time_conv_transpose, time_conv_transpose)

        # sequence pooling
        seq_pooled = self.sequence_pooling(seq_encode, seq_len)

        # text encoding
        # text_encoding = self.text_encoder(text_seq)
        # text_encoding = self.text_mlp(text_encoding)
        # text_encoding = torch.unsqueeze(text_encoding, 1)
        # seq_x = self.text_atten(torch.unsqueeze(seq_x, 1), text_encoding, text_encoding)
        # seq_x = torch.squeeze(seq_x[0], 1)

        # output
        res_cls = self.output_layers_classification(seq_pooled)
        res_reg = self.output_layers_regression(seq_pooled)
        return res_cls, torch.squeeze(res_reg, 1)

    def sequence_pooling(self, seq_x, seq_len):
        if self.pooling in ['sum', 'mean']:
            seq_x = getattr(torch, self.pooling)(seq_x, 1)
        elif self.pooling in ['max', 'min']:
            seq_x = getattr(torch, self.pooling)(seq_x, 1).values
        elif self.pooling in ['head', 'tail']:
            if self.pooling == 'head':
                seq_x = seq_x[:, 0, :]
                # idx = torch.zeros((seq_x.shape[0]),dtype=torch.long)
            else:
                raise NotImplementedError()
                # idx = seq_len.to(dtype=torch.long) - 1
                # torch.arange(len(seq_len), dtype=torch.long)

        else:
            raise NotImplementedError(f'Pooling method error: "{self.pooling}".')
        return seq_x

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        pop_labels, lev_labels, sequences, seq_len, text = batch
        # labels = labels.to(dtype=torch.long)
        probabilities, pop = self(sequences, seq_len, text)
        ce = self.loss(probabilities, lev_labels)
        self.log('Train Cross Entropy', ce, prog_bar=True)
        preds = torch.max(probabilities, keepdim=True, dim=1).indices
        precision = self.weighted_precision(preds, lev_labels)
        self.log('Train Weighted Precision', precision, prog_bar=True)

        msle = self.msle(pop, pop_labels)
        self.log('Train MSLE', msle, prog_bar=True)
        loss = ce + msle
        self.log('Train Loss', loss)
        # self.corrected_metrics(labels, preds, probabilities, seq_len)
        # loss = loss + correct_ce
        return loss

    def observed_level(self, seq_len):
        fake_labels = list(map(popularity_level, seq_len.tolist()))
        fake_labels = torch.unsqueeze(torch.tensor(fake_labels, dtype=torch.long), 1)
        return fake_labels

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        pop_labels, lev_labels, sequences, seq_len, text = batch
        # labels = labels.to(dtype=torch.long)
        probabilities, pop = self(sequences, seq_len, text)
        ce = self.loss(probabilities, lev_labels)
        self.log('Valid Cross Entropy', ce)
        preds = torch.max(probabilities, keepdim=True, dim=1).indices
        precision = self.weighted_precision(preds, lev_labels)

        msle = self.msle(pop, pop_labels)
        self.log('Valid MSLE', msle, prog_bar=False)
        loss = ce + msle
        self.log('Valid Loss', loss, )
        # self.log('Valid Weighted Precision', precision, prog_bar=False)
        # self.corrected_metrics(labels, preds, probabilities, seq_len)
        # loss = loss + correct_ce
        return {
            'loss': loss,
            'ce': ce,
            'msle': msle,
            'precision': precision,
            'labels': lev_labels,
            'preds': torch.squeeze(preds, 1),
        }

    def corrected_metrics(self, labels, preds, probabilities, seq_len):
        corrected_labels = self.observed_level(seq_len)
        corrected_labels = torch.max(torch.cat([corrected_labels, preds], 1), 1).values
        corrected_precision = self.weighted_precision(corrected_labels, labels)
        alpha = torch.sum(corrected_labels != torch.squeeze(preds, 1)) / len(seq_len)
        correct_ce = alpha * self.loss(probabilities, corrected_labels)
        self.log_metric('Correct Cross Entropy', correct_ce, prog_bar=False)
        self.log_metric('Corrected Weighted Precision', corrected_precision, prog_bar=False)

    def validation_epoch_end(self, outputs) -> None:
        result = [[], []]
        for output in outputs:
            result[0].extend(output['labels'].tolist())
            result[1].extend(output['preds'].tolist())
        labels, preds = torch.tensor(result[0]), torch.tensor(result[1])
        precision = self.weighted_precision(preds, labels)
        self.log('Valid Weighted Precision', precision, prog_bar=False, on_epoch=True)
        pass

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        pop_labels, lev_labels, sequences, seq_len, text = batch
        # labels = labels.to(dtype=torch.long)
        probabilities, pop = self(sequences, seq_len, text)
        ce = self.loss(probabilities, lev_labels)
        self.log('Test Cross Entropy', ce)
        preds = torch.max(probabilities, keepdim=True, dim=1).indices
        precision = self.weighted_precision(preds, lev_labels)
        self.log('hp_metric', precision)
        self.log('Test Weighted Precision', precision, prog_bar=True)

        msle = self.msle(pop, pop_labels)
        self.log('Test MSLE', msle, prog_bar=True)
        loss = ce + msle
        self.log('Test Loss', loss)
        return loss, precision

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None, ) -> Any:
        pop_labels, lev_labels, sequences, seq_len, text = batch
        probabilities, pop = self(sequences, seq_len, text)
        probabilities = torch.softmax(probabilities, 1)
        preds = torch.max(probabilities, keepdim=True, dim=1).indices
        preds = torch.squeeze(preds, 1)
        # observed_levels = self.observed_level(seq_len)
        # corrected weighted precision
        # result = torch.max(torch.cat([observed_levels, preds], 1), 1).values
        return preds

    def configure_optimizers(self):
        adam = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        lr_scheduler_config = {
            'optimizer': adam,
            'lr_scheduler': {
                'scheduler': CosineAnnealingWarmRestarts(adam, T_0=5, T_mult=2, verbose=True),
                'interval': 'epoch',
                # 'monitor': 'metric_to_track',
                'frequency': 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
        return lr_scheduler_config


def popularity_level(pop, shift=0):
    # 0-10为1档，11-50为2档，51-150为3档，151-300为4档，300+为5档
    if pop is None:
        return None
    levels = [11, 51, 151, 301]
    for i, lev in enumerate(levels):
        if pop < lev:
            break
    if pop >= levels[-1]:
        i += 1
    return i + shift


def collate_fn(batch):
    max_len = max(map(len, map(lambda x: x[1], batch)))
    labels = []
    text = []
    sequence = []
    seq_len = []
    for label, seq, txt in batch:
        labels.append(label)
        text.append(txt)
        t = torch.tensor(seq, dtype=torch.int)
        l = len(seq)
        seq_len.append(l)
        if l < max_len:
            zeros = torch.zeros((max_len - l, 3), dtype=torch.int)
            t = torch.cat([t, zeros], 0)
        sequence.append(t)
    try:
        # test mode
        label_levels = list(map(popularity_level, labels))
        labels = (torch.tensor(labels, dtype=torch.float), torch.tensor(label_levels, dtype=torch.long))
    except:
        labels = None, None
    # sequence.reverse()
    sequence = torch.stack(sequence, 0)
    seq_len = torch.tensor(seq_len, dtype=torch.int)
    return labels[0], labels[1], sequence, seq_len, text


def get_weight_sampler(dataset):
    labels = []
    for label, cas, text in dataset:
        labels.append(label)
    levels = [popularity_level(i) for i in labels]
    counter = Counter(levels)
    tot = len(dataset)
    p = {l: num / tot for l, num in counter.items()}

    sampler = WeightedRandomSampler([p[i] for i in levels], tot, replacement=True)
    return sampler


def train():
    show_lr_finder = True
    # training parameters
    batch_size = 64
    max_epochs = 150
    min_epochs = 20
    patience = max(min_epochs, 30)
    accumulate_grad_batches = 2
    learning_rate = 5e-4
    auto_lr_find = False
    hid_feats = 64
    smoothing = 0.
    # prepare data
    dataset = WeiboRepostDataset(mask_rate=0.1)
    train_set, valid_set, test_set = dataset.subsets()

    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn,
                              shuffle=True,
                              num_workers=8,
                              # sampler=get_weight_sampler(train_set),
                              )
    valid_loader = DataLoader(valid_set, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)
    # init model
    model = PopularityClassifier(learning_rate=learning_rate,
                                 batch_size=batch_size,
                                 hid_feats=hid_feats,
                                 pooling='head',
                                 label_smoothing=smoothing,
                                 )
    model.user_embeddings = dataset.user_profile
    # train and test
    ce_monitor = 'Valid Cross Entropy'
    loss_monitor = 'Valid Loss'
    precision_monitor = 'Valid Weighted Precision'
    early_stopping = EarlyStopping(patience=patience, monitor=loss_monitor,
                                   mode='min', verbose=True)
    model_ckpt = ModelCheckpoint(mode='min', monitor=loss_monitor, save_top_k=1, save_last=True,
                                 filename='{epoch}-{Valid Loss:.2f}',
                                 verbose=True)
    model_ckpt2 = ModelCheckpoint(mode='max', monitor=precision_monitor, save_top_k=1, save_last=True,
                                  filename='{epoch}-{Valid Weighted Precision:.2f}',
                                  verbose=True)
    trainer = pl.Trainer(max_epochs=max_epochs,
                         min_epochs=min_epochs,
                         # auto_lr_find=True,
                         # auto_scale_batch_size=True,
                         callbacks=[
                             early_stopping,
                             model_ckpt,
                             model_ckpt2,
                         ],
                         accumulate_grad_batches=accumulate_grad_batches)
    if auto_lr_find:
        lr_finder = trainer.tuner.lr_find(model,
                                          min_lr=1e-6,
                                          max_lr=5e-5,
                                          train_dataloaders=train_loader,
                                          val_dataloaders=valid_loader,
                                          )
        if show_lr_finder:
            fig = lr_finder.plot(suggest=True)
            fig.show()

        new_lr = lr_finder.suggestion()
        model.learning_rate = new_lr
        model.hparams.learning_rate = new_lr
    # trainer.tune(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader,
                ckpt_path='比赛lightning_logs/version_8/checkpoints/last.ckpt'
                )
    dataset.enhance = False
    rst = trainer.test(model, test_loader, ckpt_path='best', verbose=False)
    print(f'Test without data enhancement: {rst}')

    dataset.enhance = True
    trainer.test(model, test_loader, ckpt_path='best', verbose=False)
    print(f'Test with data enhancement: {rst}')


class ResultWriter(BasePredictionWriter):

    def __init__(self, write_interval: str = "batch", output_dir='results', id2item=None, zero_start=False) -> None:
        super().__init__(write_interval)
        self.result_file_name = f'{len(os.listdir(output_dir))}.csv'
        self.output_dir = output_dir
        self.result_file_path = osp.join(output_dir, f'{len(os.listdir(output_dir))}.csv')
        self._f = None
        self.idx = int(zero_start)
        self.id2item = id2item

    @property
    def f(self):
        if self._f is None:
            self._f = open(self.result_file_path, 'w', encoding='utf8')
            self._f.write('WeiboId\tForwardScale\n')
        return self._f

    def write(self, prediction):
        for rst in prediction.tolist():
            self.f.write(f'{self.id2item[self.idx]}\t{rst + 1}\n')
            self.idx += 1

    def write_on_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", prediction: Any,
                           batch_indices: Optional[Sequence[int]], batch: Any, batch_idx: int,
                           dataloader_idx: int) -> None:
        self.write(prediction)

    def write_on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", predictions: Sequence[Any],
                           batch_indices: Optional[Sequence[Any]]) -> None:
        for res in predictions:
            self.write(res)


def predict(ckpt, batch_size=8, output_dir='results'):
    print(f'Checkpoint: {ckpt}')
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    dataset = WeiboRepostDataset(mode='test', enhance=False)
    test_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)
    model = PopularityClassifier().load_from_checkpoint(ckpt)
    model.init_node_embeddings = dataset.user_profile
    id2item = dataset.weibo_indexer.id2item
    writer = ResultWriter(output_dir=output_dir, id2item=id2item, write_interval='batch', zero_start=dataset.zero_start)
    print(f'Predictions save in: {writer.result_file_path}')
    trainer = pl.Trainer(logger=None, callbacks=[writer])
    trainer.predict(model, test_loader)


def test(ckpt, batch_size=8):
    print(f'Checkpoint: {ckpt}')
    dataset = WeiboRepostDataset(mode='train', enhance=False)
    train_set, valid_set, test_set = dataset.subsets()
    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)
    model = PopularityClassifier().load_from_checkpoint(ckpt)
    model.init_node_embeddings = dataset.user_profile
    trainer = pl.Trainer(logger=None)
    test_result = trainer.test(model, test_loader, verbose=False)
    print(f'Test without data enhancement: {test_result}')
    dataset.enhance = True
    test_result = trainer.test(model, test_loader, verbose=False)
    print(f'Test with data enhancement: {test_result}')


if __name__ == '__main__':
    pl.seed_everything(0)
    ckpt = '比赛lightning_logs/version_9/checkpoints/epoch=146-Valid Loss=1.94.ckpt'
    # train()
    # test(ckpt)
    predict(ckpt)
