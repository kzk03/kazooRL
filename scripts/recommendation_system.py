import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import json

import torch
import torch.nn.functional as F
import yaml

from kazoo.gnn.gnn_model import GNNModel


def create_recommendation_system():
    """実用的な推薦システム"""
    
    # データ読み込み
    root = Path(__file__).resolve().parents[1]
    graph_path = root / "data/graph.pt"
    model_path = root / "data/gnn_model.pt"
    
    data = torch.load(graph_path, weights_only=False)
    model = GNNModel(in_channels_dict={"dev": 8, "task": 9}, out_channels=32)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # 埋め込み生成
    with torch.no_grad():
        embeddings = model(data.x_dict, data.edge_index_dict)
    
    def recommend_tasks_for_developer(dev_idx, top_k=10):
        """指定された開発者にタスクを推薦"""
        if dev_idx >= embeddings['dev'].size(0):
            return f"Developer {dev_idx} not found"
        
        dev_vec = embeddings['dev'][dev_idx:dev_idx+1]
        similarities = F.cosine_similarity(dev_vec, embeddings['task'])
        top_indices = torch.topk(similarities, k=min(top_k, similarities.size(0)))
        
        recommendations = []
        for rank, (sim, task_idx) in enumerate(zip(top_indices.values, top_indices.indices), 1):
            recommendations.append({
                'rank': rank,
                'task_idx': task_idx.item(),
                'similarity': sim.item()
            })
        
        return recommendations
    
    def recommend_developers_for_task(task_idx, top_k=10):
        """指定されたタスクに開発者を推薦"""
        if task_idx >= embeddings['task'].size(0):
            return f"Task {task_idx} not found"
        
        task_vec = embeddings['task'][task_idx:task_idx+1]
        similarities = F.cosine_similarity(task_vec, embeddings['dev'])
        top_indices = torch.topk(similarities, k=min(top_k, similarities.size(0)))
        
        recommendations = []
        for rank, (sim, dev_idx) in enumerate(zip(top_indices.values, top_indices.indices), 1):
            recommendations.append({
                'rank': rank,
                'dev_idx': dev_idx.item(),
                'similarity': sim.item()
            })
        
        return recommendations
    
    # デモンストレーション
    print("🤖 推薦システムデモ")
    print("\n=== 開発者0へのタスク推薦 ===")
    dev_recommendations = recommend_tasks_for_developer(0, top_k=5)
    for rec in dev_recommendations:
        print(f"{rec['rank']}. Task {rec['task_idx']}: similarity = {rec['similarity']:.4f}")
    
    print("\n=== タスク0への開発者推薦 ===")
    task_recommendations = recommend_developers_for_task(0, top_k=5)
    for rec in task_recommendations:
        print(f"{rec['rank']}. Developer {rec['dev_idx']}: similarity = {rec['similarity']:.4f}")
    
    return recommend_tasks_for_developer, recommend_developers_for_task

if __name__ == "__main__":
    create_recommendation_system()