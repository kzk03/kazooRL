# ✅ recommend_gnn.py
# 学習済みGNNモデルを使って、各devに対するtask推薦ランキングを表示

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F

from kazoo.gnn.gnn_model import GNNModel

# === パス設定
root = Path(__file__).resolve().parents[1]
graph_path = root / "data/graph.pt"
model_path = root / "data/gnn_model.pt"

# === グラフ読み込み
data = torch.load(graph_path, weights_only=False)

# === モデル初期化＆読み込み
model = GNNModel(in_channels_dict={"dev": 8, "task": 8}, out_channels=32)
model.load_state_dict(torch.load(model_path))
model.eval()

# === 推論
with torch.no_grad():
    embeddings = model(data.x_dict, data.edge_index_dict)
    dev_emb = embeddings["dev"]  # [D, dim]
    task_emb = embeddings["task"]  # [T, dim]
    scores = torch.matmul(dev_emb, task_emb.T)  # [D, T]

# === 推薦ランキング出力（Top-5）
dev_names = data["dev"].node_id

print("\n📊 GNNによるタスク推薦 (Top-5)\n")
for i, dev in enumerate(dev_names):
    topk = torch.topk(scores[i], k=5)
    print(f"▶ dev: {dev}")
    for rank, (task_idx, score) in enumerate(zip(topk.indices, topk.values), 1):
        print(f"  #{rank}: task_{task_idx.item()} (score: {score.item():.4f})")
    print()
