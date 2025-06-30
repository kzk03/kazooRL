import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GNNModel(nn.Module):
    """
    GATConv を用いた異種グラフニューラルネットワークモデル。
    開発者-タスク間の関係に加えて、開発者-開発者間の協力関係もモデル化。
    """

    def __init__(self, in_channels_dict, out_channels, hidden_channels=16, heads=4):
        super().__init__()

        # 第1層の出力次元 = hidden_channels * heads
        layer1_out_channels = hidden_channels * heads  # 16 * 4 = 64

        # レイヤー1: 異種ノード間（開発者-タスク）
        self.gat_dev_writes_task_1 = GATConv(
            (in_channels_dict["dev"], in_channels_dict["task"]),
            hidden_channels,
            heads=heads,
            add_self_loops=False,
        )

        self.gat_task_written_by_dev_1 = GATConv(
            (in_channels_dict["task"], in_channels_dict["dev"]),
            hidden_channels,
            heads=heads,
            add_self_loops=False,
        )

        # 🆕 開発者間の協力関係レイヤー
        self.gat_dev_collaborates_dev_1 = GATConv(
            in_channels_dict["dev"],  # 同種ノード間
            hidden_channels,
            heads=heads,
            add_self_loops=False,
        )

        # レイヤー2: 異種ノード間（第1層の出力を使用）
        self.gat_dev_writes_task_2 = GATConv(
            (layer1_out_channels, layer1_out_channels),  # 両方とも64次元
            out_channels,
            heads=1,
            add_self_loops=False,
        )

        self.gat_task_written_by_dev_2 = GATConv(
            (layer1_out_channels, layer1_out_channels),  # 両方とも64次元
            out_channels,
            heads=1,
            add_self_loops=False,
        )

        # 🆕 第2層の開発者間協力関係
        self.gat_dev_collaborates_dev_2 = GATConv(
            layer1_out_channels,  # 第1層の出力次元
            out_channels,
            heads=1,
            add_self_loops=False,
        )

    def _clean_edge_index(self, edge_index, max_src_nodes, max_dst_nodes, edge_name=""):
        """エッジインデックスを検証・クリーニングする"""
        if edge_index.size(1) == 0:
            return edge_index

        # 有効な範囲内のエッジのみを保持
        src_valid = (edge_index[0] >= 0) & (edge_index[0] < max_src_nodes)
        dst_valid = (edge_index[1] >= 0) & (edge_index[1] < max_dst_nodes)
        valid_mask = src_valid & dst_valid

        if not valid_mask.all():
            invalid_count = (~valid_mask).sum().item()
            print(f"Warning: Removing {invalid_count} invalid edges from {edge_name}")
            edge_index = edge_index[:, valid_mask]

        return edge_index

    def forward(self, x_dict, edge_index_dict):
        # ノード数を取得
        num_dev_nodes = x_dict["dev"].size(0)
        num_task_nodes = x_dict["task"].size(0)

        # 基本的な妥当性チェック
        if num_dev_nodes == 0 or num_task_nodes == 0:
            return {
                "dev": torch.zeros((num_dev_nodes, 32), dtype=torch.float32),
                "task": torch.zeros((num_task_nodes, 32), dtype=torch.float32),
            }

        # エッジをクリーニング
        clean_edge_dict = {}

        # dev -> task エッジをクリーニング
        clean_edge_dict[("dev", "writes", "task")] = self._clean_edge_index(
            edge_index_dict[("dev", "writes", "task")],
            num_dev_nodes,
            num_task_nodes,
            "dev->task writes",
        )

        # task -> dev エッジをクリーニング
        clean_edge_dict[("task", "written_by", "dev")] = self._clean_edge_index(
            edge_index_dict[("task", "written_by", "dev")],
            num_task_nodes,
            num_dev_nodes,
            "task->dev written_by",
        )

        # 🆕 開発者間協力関係エッジをクリーニング（存在する場合）
        if ("dev", "collaborates", "dev") in edge_index_dict:
            clean_edge_dict[("dev", "collaborates", "dev")] = self._clean_edge_index(
                edge_index_dict[("dev", "collaborates", "dev")],
                num_dev_nodes,
                num_dev_nodes,
                "dev->dev collaborates",
            )

        # エッジが存在しない場合の処理
        if clean_edge_dict[("dev", "writes", "task")].size(1) == 0:
            print("Warning: No valid edges found. Returning zero embeddings.")
            return {
                "dev": torch.zeros((num_dev_nodes, 32), dtype=torch.float32),
                "task": torch.zeros((num_task_nodes, 32), dtype=torch.float32),
            }

        try:
            # 第1層: dev -> task
            x_task_1 = self.gat_dev_writes_task_1(
                (x_dict["dev"], x_dict["task"]),
                clean_edge_dict[("dev", "writes", "task")],
            )
            x_task_1 = F.relu(x_task_1)

            # 第1層: task -> dev
            x_dev_1 = self.gat_task_written_by_dev_1(
                (x_dict["task"], x_dict["dev"]),
                clean_edge_dict[("task", "written_by", "dev")],
            )
            x_dev_1 = F.relu(x_dev_1)

            # 🆕 第1層: 開発者間協力関係
            x_dev_collab_1 = x_dev_1  # デフォルト値
            if ("dev", "collaborates", "dev") in clean_edge_dict:
                collab_edges = clean_edge_dict[("dev", "collaborates", "dev")]
                if collab_edges.size(1) > 0:
                    x_dev_collab_1 = self.gat_dev_collaborates_dev_1(
                        x_dict["dev"], collab_edges
                    )
                    x_dev_collab_1 = F.relu(x_dev_collab_1)

            # 開発者特徴量を協力関係の情報と結合
            x_dev_combined = x_dev_1 + 0.5 * x_dev_collab_1  # 重み付き結合

            # 第2層: dev -> task（協力関係を考慮した開発者特徴量を使用）
            x_task_2 = self.gat_dev_writes_task_2(
                (x_dev_combined, x_task_1),
                clean_edge_dict[("dev", "writes", "task")],
            )
            x_task_2 = F.relu(x_task_2)

            # 第2層: task -> dev（協力関係を考慮した開発者特徴量を使用）
            x_dev_2 = self.gat_task_written_by_dev_2(
                (x_task_1, x_dev_combined),
                clean_edge_dict[("task", "written_by", "dev")],
            )
            x_dev_2 = F.relu(x_dev_2)

            # 🆕 第2層: 開発者間協力関係（オプション）
            if ("dev", "collaborates", "dev") in clean_edge_dict:
                collab_edges = clean_edge_dict[("dev", "collaborates", "dev")]
                if collab_edges.size(1) > 0:
                    x_dev_collab_2 = self.gat_dev_collaborates_dev_2(
                        x_dev_combined, collab_edges
                    )
                    x_dev_collab_2 = F.relu(x_dev_collab_2)
                    # 最終的な開発者特徴量に協力関係の情報を統合
                    x_dev_2 = x_dev_2 + 0.3 * x_dev_collab_2

            return {"dev": x_dev_2, "task": x_task_2}

        except Exception as e:
            print(f"Error in GNN layers: {e}")
            print("Returning zero embeddings as fallback")
            return {
                "dev": torch.zeros((num_dev_nodes, 32), dtype=torch.float32),
                "task": torch.zeros((num_task_nodes, 32), dtype=torch.float32),
            }
