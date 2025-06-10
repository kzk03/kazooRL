import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class GNNModel(nn.Module):
    def __init__(self, in_channels_dict, out_channels):
        super().__init__()

        self.convs1 = nn.ModuleDict(
            {
                "writes": SAGEConv(in_channels_dict["dev"], 16),
                "reviews": SAGEConv(in_channels_dict["dev"], 16),
                "written_by": SAGEConv(in_channels_dict["task"], 16),
                "reviewed_by": SAGEConv(in_channels_dict["task"], 16),
            }
        )

        self.convs2 = nn.ModuleDict(
            {
                "writes": SAGEConv(16, out_channels),
                "reviews": SAGEConv(16, out_channels),
                "written_by": SAGEConv(16, out_channels),
                "reviewed_by": SAGEConv(16, out_channels),
            }
        )

        self.conv1 = HeteroConv(
            {
                ("dev", "writes", "task"): self.convs1["writes"],
                ("dev", "reviews", "task"): self.convs1["reviews"],
                ("task", "written_by", "dev"): self.convs1["written_by"],
                ("task", "reviewed_by", "dev"): self.convs1["reviewed_by"],
            },
            aggr="sum",
        )

        self.conv2 = HeteroConv(
            {
                ("dev", "writes", "task"): self.convs2["writes"],
                ("dev", "reviews", "task"): self.convs2["reviews"],
                ("task", "written_by", "dev"): self.convs2["written_by"],
                ("task", "reviewed_by", "dev"): self.convs2["reviewed_by"],
            },
            aggr="sum",
        )

    def forward(self, x_dict, edge_index_dict):
        # 正しい呼び方（dict形式でそのまま渡す）
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items() if v is not None}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict
