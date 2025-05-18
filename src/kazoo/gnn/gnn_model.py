import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class GNNModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = HeteroConv({
            ('dev', 'writes', 'task'): SAGEConv(in_channels, 16),
            ('dev', 'reviews', 'task'): SAGEConv(in_channels, 16),
            ('task', 'written_by', 'dev'): SAGEConv(in_channels, 16),
            ('task', 'reviewed_by', 'dev'): SAGEConv(in_channels, 16)
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('dev', 'writes', 'task'): SAGEConv(16, out_channels),
            ('dev', 'reviews', 'task'): SAGEConv(16, out_channels),
            ('task', 'written_by', 'dev'): SAGEConv(16, out_channels),
            ('task', 'reviewed_by', 'dev'): SAGEConv(16, out_channels)
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items() if v is not None}  # ← 安全処理
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict