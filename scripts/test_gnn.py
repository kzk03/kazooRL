
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F

from kazoo.gnn.gnn_model import GNNModel


def test_gnn_predictions():
    """学習済みGNNモデルで予測を実行"""
    
    # パス設定
    root = Path(__file__).resolve().parents[1]
    graph_path = root / "data/graph.pt"
    model_path = root / "data/gnn_model.pt"
    
    # データ読み込み
    print("Loading data...")
    data = torch.load(graph_path, weights_only=False)
    
    # モデル初期化と重み読み込み
    print("Loading trained model...")
    model = GNNModel(
        in_channels_dict={"dev": 8, "task": 9}, 
        out_channels=32
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # 予測実行
    print("Running predictions...")
    with torch.no_grad():
        embeddings = model(data.x_dict, data.edge_index_dict)
    
    print(f"✅ Embeddings generated:")
    print(f"  - Developer embeddings: {embeddings['dev'].shape}")
    print(f"  - Task embeddings: {embeddings['task'].shape}")
    
    # 類似度計算の例
    dev_emb = embeddings['dev']
    task_emb = embeddings['task']
    
    # 最初の開発者と最初の10タスクの類似度
    if dev_emb.size(0) > 0 and task_emb.size(0) >= 10:
        dev_0 = dev_emb[0:1]  # 形状: [1, 32]
        task_0_9 = task_emb[0:10]  # 形状: [10, 32]
        
        similarities = F.cosine_similarity(dev_0, task_0_9)
        print(f"\n開発者0と最初の10タスクの類似度:")
        for i, sim in enumerate(similarities):
            print(f"  Task {i}: {sim.item():.4f}")
    
    # 推薦例：開発者0に最適なタスクを探す
    if dev_emb.size(0) > 0 and task_emb.size(0) > 0:
        dev_0 = dev_emb[0:1]  # 形状: [1, 32]
        all_similarities = F.cosine_similarity(dev_0, task_emb)
        
        # トップ5を取得
        top_5_indices = torch.topk(all_similarities, k=min(5, task_emb.size(0))).indices
        print(f"\n開発者0に推薦するトップ5タスク:")
        for i, task_idx in enumerate(top_5_indices):
            sim = all_similarities[task_idx].item()
            print(f"  {i+1}. Task {task_idx.item()}: similarity = {sim:.4f}")

if __name__ == "__main__":
    test_gnn_predictions()