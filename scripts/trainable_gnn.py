import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import random

import torch
import torch.nn.functional as F
from torch.optim import Adam

from kazoo.gnn.gnn_model import GNNModel


def create_better_training_data(data):
    """より良い学習データを作成"""
    
    # 実際のエッジ関係を取得
    writes_edges = data[('dev', 'writes', 'task')].edge_index
    num_dev = data['dev'].x.size(0)
    num_task = data['task'].x.size(0)
    
    # 有効なエッジのみ使用
    valid_mask = (writes_edges[0] < num_dev) & (writes_edges[1] < num_task)
    valid_edges = writes_edges[:, valid_mask]
    
    if valid_edges.size(1) == 0:
        print("Error: No valid edges found")
        return None, None
    
    # 正のサンプル（実際の関係）
    pos_pairs = valid_edges.t()  # [num_edges, 2]
    pos_labels = torch.ones(pos_pairs.size(0))
    
    print(f"正のサンプル数: {pos_pairs.size(0)}")
    
    # ▼▼▼【改善点】より戦略的な負のサンプル生成▼▼▼
    neg_pairs = []
    neg_labels = []
    
    # 実際の関係をセットに変換（高速検索のため）
    actual_pairs = set()
    for i in range(pos_pairs.size(0)):
        actual_pairs.add((pos_pairs[i, 0].item(), pos_pairs[i, 1].item()))
    
    # 1. 完全ランダムな負のサンプル
    random_neg_count = pos_pairs.size(0) // 2
    random_neg_pairs = []
    for _ in range(random_neg_count * 2):  # 余分に生成してフィルタリング
        dev_idx = random.randint(0, num_dev - 1)
        task_idx = random.randint(0, num_task - 1)
        if (dev_idx, task_idx) not in actual_pairs:
            random_neg_pairs.append([dev_idx, task_idx])
            if len(random_neg_pairs) >= random_neg_count:
                break
    
    # 2. 部分的に関連する負のサンプル（同じ開発者、異なるタスク）
    partial_neg_count = pos_pairs.size(0) // 4
    partial_neg_pairs = []
    for i in range(min(partial_neg_count * 3, pos_pairs.size(0))):
        dev_idx = pos_pairs[i % pos_pairs.size(0), 0].item()
        # ランダムなタスクを選択
        for _ in range(10):  # 最大10回試行
            task_idx = random.randint(0, num_task - 1)
            if (dev_idx, task_idx) not in actual_pairs:
                partial_neg_pairs.append([dev_idx, task_idx])
                break
        if len(partial_neg_pairs) >= partial_neg_count:
            break
    
    # 3. 人気タスクでの負のサンプル（推薦されやすいタスクでの誤分類を防ぐ）
    # タスクの人気度を計算
    task_popularity = torch.zeros(num_task)
    for i in range(pos_pairs.size(0)):
        task_idx = pos_pairs[i, 1].item()
        task_popularity[task_idx] += 1
    
    # 人気上位20%のタスクを取得
    top_k = max(1, int(0.2 * num_task))
    popular_tasks = torch.topk(task_popularity, k=top_k).indices
    
    popular_neg_count = pos_pairs.size(0) // 4
    popular_neg_pairs = []
    for _ in range(popular_neg_count * 2):
        dev_idx = random.randint(0, num_dev - 1)
        task_idx = popular_tasks[random.randint(0, len(popular_tasks) - 1)].item()
        if (dev_idx, task_idx) not in actual_pairs:
            popular_neg_pairs.append([dev_idx, task_idx])
            if len(popular_neg_pairs) >= popular_neg_count:
                break
    
    # 負のサンプルを結合
    all_neg_pairs = random_neg_pairs + partial_neg_pairs + popular_neg_pairs
    
    if not all_neg_pairs:
        print("Warning: No negative samples generated, using random fallback")
        # フォールバック: 完全ランダム
        neg_count = pos_pairs.size(0)
        for _ in range(neg_count * 3):
            dev_idx = random.randint(0, num_dev - 1)
            task_idx = random.randint(0, num_task - 1)
            if (dev_idx, task_idx) not in actual_pairs:
                all_neg_pairs.append([dev_idx, task_idx])
                if len(all_neg_pairs) >= neg_count:
                    break
    
    neg_pairs_tensor = torch.tensor(all_neg_pairs[:len(all_neg_pairs)])
    neg_labels = torch.zeros(neg_pairs_tensor.size(0))
    
    print(f"負のサンプル数: {neg_pairs_tensor.size(0)} (ランダム:{len(random_neg_pairs)}, 部分:{len(partial_neg_pairs)}, 人気:{len(popular_neg_pairs)})")
    # ▲▲▲【改善点ここまで】▲▲▲
    
    # 最終的なデータセットを作成
    all_pairs = torch.cat([pos_pairs, neg_pairs_tensor], dim=0)
    all_labels = torch.cat([pos_labels, neg_labels], dim=0)
    
    # データをシャッフル
    indices = torch.randperm(all_pairs.size(0))
    all_pairs = all_pairs[indices]
    all_labels = all_labels[indices]
    
    return all_pairs, all_labels

def main():
    # パス設定
    root = Path(__file__).resolve().parents[1]
    graph_path = root / "data/graph.pt"
    
    # データ読み込み
    print("Loading graph data...")
    data = torch.load(graph_path, weights_only=False)
    print(f"Graph data loaded: dev={data['dev'].x.shape}, task={data['task'].x.shape}")
    
    # 改良された学習データを作成
    print("Creating improved training data...")
    pairs, labels = create_better_training_data(data)
    
    if pairs is None:
        print("Failed to create training data")
        return
    
    print(f"Training data created: {pairs.size(0)} pairs")
    
    # 次元数を正しく設定
    actual_dev_dim = data['dev'].x.shape[1]
    actual_task_dim = data['task'].x.shape[1]
    
    # モデル初期化
    print("Initializing model...")
    model = GNNModel(in_channels_dict={"dev": actual_dev_dim, "task": actual_task_dim}, out_channels=32)
    optimizer = Adam(model.parameters(), lr=5e-4)
    
    print("Starting improved training...")
    
    # 学習ループ
    for epoch in range(1, 21):
        model.train()
        
        try:
            embeddings = model(data.x_dict, data.edge_index_dict)
            
            # ペアの妥当性をチェック
            valid_pairs_mask = (pairs[:, 0] < embeddings["dev"].size(0)) & (pairs[:, 1] < embeddings["task"].size(0))
            valid_pairs = pairs[valid_pairs_mask]
            valid_labels = labels[valid_pairs_mask]
            
            if valid_pairs.size(0) == 0:
                print("No valid pairs found. Skipping epoch.")
                continue
            
            dev_emb = embeddings["dev"][valid_pairs[:, 0]]
            task_emb = embeddings["task"][valid_pairs[:, 1]]
            
            # L2正規化を追加
            dev_emb = F.normalize(dev_emb, p=2, dim=1)
            task_emb = F.normalize(task_emb, p=2, dim=1)
            
            # 内積ベースのスコア
            scores = torch.sum(dev_emb * task_emb, dim=1)
            
            target_labels = valid_labels.float()
            loss = F.binary_cross_entropy_with_logits(scores, target_labels)
            
            # L2正則化を追加
            l2_reg = 0.01 * (torch.norm(dev_emb) + torch.norm(task_emb))
            total_loss = loss + l2_reg
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0:
                print(f"[Epoch {epoch:03d}] loss: {loss.item():.4f}, total_loss: {total_loss.item():.4f}, valid_pairs: {valid_pairs.size(0)}")
            
        except Exception as e:
            print(f"Error in epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # ▼▼▼【変更箇所】元のファイル名で保存▼▼▼
    model_save_path = root / "data/gnn_model.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ 改良版モデルを保存しました → {model_save_path}")
    # ▲▲▲【変更箇所ここまで】▲▲▲

if __name__ == "__main__":
    main()