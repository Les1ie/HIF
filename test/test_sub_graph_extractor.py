import os
import os.path as osp

import matplotlib.pyplot as plt
import networkx as nx
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.optim import Adam
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GNNExplainer
from torch_geometric.utils import to_networkx

class Net(pl.LightningModule):

    def __init__(self, in_feats, hid_feats, out_feats, learning_rate=1e-2, weight_decay=5e-4):
        super(Net, self).__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats

        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats, out_feats)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def configure_optimizers(self):
        adam = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return adam

    def training_step(self, *args, **kwargs):
        x, y = args
        # loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
        return


if __name__ == '__main__':
    dataset = 'Cora'
    path = osp.join(os.getcwd(), 'data', 'Planetoid')
    print('data dir: {}'.format(path))
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset.num_features, 16, dataset.num_classes).to(device)
    data = data.to(device)
    x, edge_index = data.x, data.edge_index

    # dataloader = DataLoader()
    optimizer = model.configure_optimizers()
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        log_logits = model(x, edge_index)
        loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    explainer = GNNExplainer(model, epochs=200,
                             # return_type='log_prob'
                             )
    node_idx = 10
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
    ax, G = explainer.visualize_subgraph(node_idx, edge_index, edge_mask, y=data.y)
    plt.show()
    g = to_networkx(data)
    nx.draw(g, node_size=10, with_labels=True)
    plt.show()
