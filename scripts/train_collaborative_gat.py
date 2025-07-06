#!/usr/bin/env python3
"""
協力ネットワーク対応の新しいGATモデルを訓練・保存するスクリプト
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

# プロジェクトルートを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kazoo.GAT.GAT_model import GATModel


def train_collaborative_gat():
    """協力ネットワーク対応のGATモデルを訓練"""

    print("=== Training Collaborative GAT Model ===")

    # グラフデータをロード
    graph_path = Path("data/graph_training.pt")
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
    model = GATModel(in_channels_dict=in_channels_dict, out_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"Model architecture:\n{model}")

    # 訓練ループ
    print("\n=== Training Loop ===")
    model.train()

    epoch_progress = tqdm(
        range(200),  # 本格的な訓練のため200エポック（50→200）
        desc="🧠 GAT 訓練",
        unit="epoch",
        colour="cyan",
        leave=True,
    )

    for epoch in epoch_progress:
        optimizer.zero_grad()

        try:
            # フォワードパス
            embeddings = model(data.x_dict, edge_index_dict)

            # 1. リンク予測損失（dev-task エッジ）
            dev_task_edges = edge_index_dict[("dev", "writes", "task")]
            if dev_task_edges.size(1) > 0:
                # 正例: 実際に存在するエッジ
                src_embeds = embeddings["dev"][dev_task_edges[0]]
                dst_embeds = embeddings["task"][dev_task_edges[1]]
                pos_scores = F.cosine_similarity(src_embeds, dst_embeds)

                # 負例: ランダムに選んだ存在しないエッジ
                num_neg = min(dev_task_edges.size(1), 100)  # 負例数を制限
                neg_dev_idx = torch.randint(0, embeddings["dev"].size(0), (num_neg,))
                neg_task_idx = torch.randint(0, embeddings["task"].size(0), (num_neg,))
                neg_src_embeds = embeddings["dev"][neg_dev_idx]
                neg_dst_embeds = embeddings["task"][neg_task_idx]
                neg_scores = F.cosine_similarity(neg_src_embeds, neg_dst_embeds)

                # バイナリクロスエントロピー損失
                pos_loss = F.binary_cross_entropy_with_logits(
                    pos_scores, torch.ones_like(pos_scores)
                )
                neg_loss = F.binary_cross_entropy_with_logits(
                    neg_scores, torch.zeros_like(neg_scores)
                )
                link_loss = pos_loss + neg_loss
            else:
                link_loss = torch.tensor(0.0)

            # 2. 協力ネットワークエッジに基づく損失
            collab_edges = edge_index_dict[("dev", "collaborates", "dev")]
            if collab_edges.size(1) > 0:
                # 協力している開発者ペアの埋め込みを近づける
                src_embeds = embeddings["dev"][collab_edges[0]]
                dst_embeds = embeddings["dev"][collab_edges[1]]
                collab_similarity = F.cosine_similarity(src_embeds, dst_embeds)
                collab_loss = -collab_similarity.mean()  # 類似度を最大化
            else:
                collab_loss = torch.tensor(0.0)

            # 3. 埋め込み正則化損失（L2正則化）
            dev_reg = torch.norm(embeddings["dev"], p=2, dim=1).mean()
            task_reg = torch.norm(embeddings["task"], p=2, dim=1).mean()
            reg_loss = 0.001 * (dev_reg + task_reg)

            # 総損失
            total_loss = link_loss + 0.1 * collab_loss + reg_loss

            total_loss.backward()
            optimizer.step()

            # 進捗バーの情報更新
            epoch_progress.set_postfix(
                {
                    "Loss": f"{total_loss.item():.4f}",
                    "Link": f"{link_loss.item():.4f}",
                    "Collab": f"{collab_loss.item():.4f}",
                    "Reg": f"{reg_loss.item():.4f}",
                }
            )

            if (epoch + 1) % 20 == 0:
                print(
                    f"\nEpoch {epoch+1:3d}: Loss = {total_loss.item():.4f} "
                    f"(link: {link_loss.item():.4f}, collab: {collab_loss.item():.4f}, reg: {reg_loss.item():.4f})"
                )

        except Exception as e:
            epoch_progress.set_postfix({"Error": str(e)[:20]})
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
