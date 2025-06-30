#!/usr/bin/env python3
"""
協力ネットワーク対応の新しいGATモデルを訓練・保存するスクリプト
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# プロジェクトルートを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kazoo.GAT.GAT_model import GNNModel


def train_collaborative_gat():
    """協力ネットワーク対応のGATモデルを訓練"""

    print("=== Training Collaborative GAT Model ===")

    # グラフデータをロード
    graph_path = Path("data/graph.pt")
    if not graph_path.exists():
        print(f"Error: Graph data not found: {graph_path}")
        return

    data = torch.load(graph_path, weights_only=False)
    print(f"Graph loaded: {data}")

    # 開発者協力ネットワークをロード
    network_path = Path("data/developer_collaboration_network.pt")
    if not network_path.exists():
        print(f"Error: Developer collaboration network not found: {network_path}")
        return

    dev_network = torch.load(network_path, weights_only=False)
    print(
        f"Collaboration network loaded: {dev_network['num_developers']} developers, {dev_network['network_stats']['num_edges']} edges"
    )

    # エッジ辞書を準備（協力ネットワークを含む）
    edge_index_dict = {
        ("dev", "writes", "task"): data[("dev", "writes", "task")].edge_index,
        ("task", "written_by", "dev"): data[("dev", "writes", "task")].edge_index.flip(
            [0]
        ),
        ("dev", "collaborates", "dev"): dev_network["dev_collaboration_edge_index"],
    }

    print("Edge types:")
    for edge_type, edge_index in edge_index_dict.items():
        print(f"  {edge_type}: {edge_index.shape}")

    # モデル初期化
    in_channels_dict = {"dev": 8, "task": 9}
    model = GNNModel(in_channels_dict=in_channels_dict, out_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"Model architecture:\n{model}")

    # 訓練ループ
    print("\n=== Training Loop ===")
    model.train()

    for epoch in range(100):  # 100エポック訓練
        optimizer.zero_grad()

        try:
            # フォワードパス
            embeddings = model(data.x_dict, edge_index_dict)

            # 簡単な自己教師学習損失（ノード埋め込みの一貫性）
            dev_loss = F.mse_loss(embeddings["dev"], embeddings["dev"].detach())
            task_loss = F.mse_loss(embeddings["task"], embeddings["task"].detach())

            # 協力ネットワークエッジに基づく損失
            collab_edges = edge_index_dict[("dev", "collaborates", "dev")]
            if collab_edges.size(1) > 0:
                # 協力している開発者ペアの埋め込みを近づける
                src_embeds = embeddings["dev"][collab_edges[0]]
                dst_embeds = embeddings["dev"][collab_edges[1]]
                collab_loss = -F.cosine_similarity(src_embeds, dst_embeds).mean()
            else:
                collab_loss = torch.tensor(0.0)

            total_loss = dev_loss + task_loss + 0.1 * collab_loss

            total_loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch+1:3d}: Loss = {total_loss.item():.4f} "
                    f"(dev: {dev_loss.item():.4f}, task: {task_loss.item():.4f}, collab: {collab_loss.item():.4f})"
                )

        except Exception as e:
            print(f"Error in epoch {epoch}: {e}")
            break

    # モデルを保存
    model.eval()

    # 最終的な埋め込みを計算
    with torch.no_grad():
        final_embeddings = model(data.x_dict, edge_index_dict)

    # モデル保存
    model_save_path = "data/gnn_model_collaborative.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Collaborative GAT model saved: {model_save_path}")

    # 更新されたグラフデータも保存（協力ネットワークエッジを含む）
    enhanced_data = data.clone()
    enhanced_data[("dev", "collaborates", "dev")].edge_index = dev_network[
        "dev_collaboration_edge_index"
    ]
    enhanced_data[("dev", "collaborates", "dev")].edge_attr = dev_network[
        "dev_collaboration_edge_weights"
    ]

    graph_save_path = "data/graph_collaborative.pt"
    torch.save(enhanced_data, graph_save_path)
    print(f"✅ Enhanced graph data saved: {graph_save_path}")

    print("\n=== Training Completed ===")
    print(f"Final embeddings:")
    print(f"  - Developer embeddings: {final_embeddings['dev'].shape}")
    print(f"  - Task embeddings: {final_embeddings['task'].shape}")


if __name__ == "__main__":
    train_collaborative_gat()
