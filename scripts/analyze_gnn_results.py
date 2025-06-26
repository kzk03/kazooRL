import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from kazoo.gnn.gnn_model import GNNModel


def load_metadata():
    """開発者とタスクのメタデータを読み込み"""
    
    # 開発者プロファイル読み込み
    profile_path = Path("configs/dev_profiles.yaml")
    if profile_path.exists():
        import yaml
        with open(profile_path, 'r', encoding='utf-8') as f:
            profiles = yaml.safe_load(f)
    else:
        profiles = None
    
    # タスクメタデータ読み込み（JSON形式）
    status_dir = Path("data/status/")
    task_metadata = []
    
    if status_dir.exists():
        for json_file in status_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            task_data = json.loads(line)
                            task_metadata.append(task_data)
            except Exception as e:
                print(f"Warning: Could not read {json_file}: {e}")
    
    return profiles, task_metadata

def analyze_gnn_results():
    """GNN結果の詳細分析"""
    
    # パス設定
    root = Path(__file__).resolve().parents[1]
    graph_path = root / "data/graph.pt"
    model_path = root / "data/gnn_model.pt"
    
    # データ読み込み
    print("Loading data...")
    data = torch.load(graph_path, weights_only=False)
    
    # メタデータ読み込み
    profiles, task_metadata = load_metadata()
    
    # モデル読み込み
    print("Loading model...")
    model = GNNModel(in_channels_dict={"dev": 8, "task": 9}, out_channels=32)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # 予測実行
    print("Running analysis...")
    with torch.no_grad():
        embeddings = model(data.x_dict, data.edge_index_dict)
    
    dev_emb = embeddings['dev']
    task_emb = embeddings['task']
    
    # ▼▼▼【修正箇所】node_idの適切な処理▼▼▼
    # 開発者IDとタスクIDを取得
    if hasattr(data['dev'], 'node_id'):
        if isinstance(data['dev'].node_id, torch.Tensor):
            dev_ids = data['dev'].node_id.tolist()
        elif isinstance(data['dev'].node_id, list):
            dev_ids = data['dev'].node_id
        else:
            dev_ids = list(range(dev_emb.size(0)))
    else:
        dev_ids = list(range(dev_emb.size(0)))
    
    if hasattr(data['task'], 'node_id'):
        if isinstance(data['task'].node_id, torch.Tensor):
            task_ids = data['task'].node_id.tolist()
        elif isinstance(data['task'].node_id, list):
            task_ids = data['task'].node_id
        else:
            task_ids = list(range(task_emb.size(0)))
    else:
        task_ids = list(range(task_emb.size(0)))
    # ▲▲▲【修正箇所ここまで】▲▲▲
    
    print(f"\n=== 詳細分析結果 ===")
    print(f"開発者数: {len(dev_ids)}, タスク数: {len(task_ids)}")
    
    # 1. 類似度分布の分析
    print("\n📊 類似度分布の分析")
    dev_0 = dev_emb[0:1]
    all_similarities = F.cosine_similarity(dev_0, task_emb)
    unique_values, counts = torch.unique(all_similarities.round(decimals=4), return_counts=True)
    
    print(f"類似度の値の種類: {len(unique_values)}")
    for val, count in zip(unique_values[:10], counts[:10]):  # 上位10個を表示
        print(f"  {val.item():.4f}: {count.item()}件")
    
    # 2. 開発者別の推薦分析
    print(f"\n🎯 開発者別推薦分析（上位5名）")
    for dev_idx in range(min(5, dev_emb.size(0))):
        dev_vec = dev_emb[dev_idx:dev_idx+1]
        similarities = F.cosine_similarity(dev_vec, task_emb)
        top_5 = torch.topk(similarities, k=5)
        
        dev_id = dev_ids[dev_idx] if dev_idx < len(dev_ids) else 'Unknown'
        print(f"\n開発者 {dev_idx} (ID: {dev_id}):")
        for rank, (sim, task_idx) in enumerate(zip(top_5.values, top_5.indices), 1):
            task_id = task_ids[task_idx] if task_idx < len(task_ids) else 'Unknown'
            print(f"  {rank}. Task {task_idx.item()} (ID: {task_id}): {sim.item():.4f}")
    
    # 3. タスククラスタリング分析
    print(f"\n🔍 タスククラスタリング分析")
    # タスク埋め込みの類似度行列を計算（計算量を考慮してサンプリング）
    sample_size = min(100, task_emb.size(0))
    sample_indices = torch.randperm(task_emb.size(0))[:sample_size]
    sample_task_emb = task_emb[sample_indices]
    
    # 類似度行列計算
    task_similarity_matrix = F.cosine_similarity(
        sample_task_emb.unsqueeze(1), 
        sample_task_emb.unsqueeze(0), 
        dim=2
    )
    
    # 高い類似度を持つタスクペアを探索
    upper_triangle = torch.triu(task_similarity_matrix, diagonal=1)
    high_sim_indices = torch.where(upper_triangle > 0.8)
    
    print(f"高い類似度（>0.8）を持つタスクペア: {len(high_sim_indices[0])}組")
    for i in range(min(5, len(high_sim_indices[0]))):
        idx1, idx2 = high_sim_indices[0][i], high_sim_indices[1][i]
        sim = task_similarity_matrix[idx1, idx2].item()
        task1_idx = sample_indices[idx1].item()
        task2_idx = sample_indices[idx2].item()
        task1_id = task_ids[task1_idx] if task1_idx < len(task_ids) else 'Unknown'
        task2_id = task_ids[task2_idx] if task2_idx < len(task_ids) else 'Unknown'
        print(f"  Task {task1_idx} (ID: {task1_id}) - Task {task2_idx} (ID: {task2_id}): {sim:.4f}")
    
    # 4. 開発者の専門性分析
    print(f"\n👥 開発者の専門性分析")
    # 各開発者について、最も類似度の高いタスクの特徴を分析
    for dev_idx in range(min(3, dev_emb.size(0))):
        dev_vec = dev_emb[dev_idx:dev_idx+1]
        similarities = F.cosine_similarity(dev_vec, task_emb)
        
        # 上位20%のタスクを専門分野として定義
        top_k = max(1, int(0.2 * task_emb.size(0)))
        top_tasks = torch.topk(similarities, k=top_k).indices
        
        mean_sim = similarities[top_tasks].mean().item()
        std_sim = similarities[top_tasks].std().item()
        
        dev_id = dev_ids[dev_idx] if dev_idx < len(dev_ids) else 'Unknown'
        print(f"開発者 {dev_idx} (ID: {dev_id}):")
        print(f"  専門分野タスク数: {top_k}")
        print(f"  平均類似度: {mean_sim:.4f}")
        print(f"  類似度標準偏差: {std_sim:.4f}")
    
    # 5. 全体の統計情報
    print(f"\n📈 全体統計")
    # 全ペアの類似度統計
    all_dev_similarities = []
    for dev_idx in range(min(10, dev_emb.size(0))):  # 計算量削減のため10名まで
        dev_vec = dev_emb[dev_idx:dev_idx+1]
        similarities = F.cosine_similarity(dev_vec, task_emb)
        all_dev_similarities.append(similarities)
    
    if all_dev_similarities:
        all_sims = torch.cat(all_dev_similarities)
        print(f"類似度統計（サンプル10名分）:")
        print(f"  平均: {all_sims.mean().item():.4f}")
        print(f"  標準偏差: {all_sims.std().item():.4f}")
        print(f"  最小値: {all_sims.min().item():.4f}")
        print(f"  最大値: {all_sims.max().item():.4f}")

if __name__ == "__main__":
    analyze_gnn_results()