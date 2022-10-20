import logging

logger = logging.getLogger(__name__)

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean


class ChargeModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_weight

        x = x.view(-1, 1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        x = scatter_mean(x.view(-1, 1), data.batch, dim=0)

        x = x.view(len(x))

        return x
