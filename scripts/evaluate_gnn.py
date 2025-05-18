import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import torch
from sklearn.metrics import roc_auc_score

from kazoo.gnn.gnn_model import GNNModel

# === パス設定
root = Path(__file__).resolve().parents[1]
graph_path = root / "data/graph.pt"
label_path = root / "data/labels.pt"
model_path = root / "data/gnn_model.pt"

# === データ読み込み
data = torch.load(graph_path, weights_only=False)
pairs, labels = torch.load(label_path)

# === モデル読み込み
model = GNNModel(in_channels_dict={"dev": 8, "task": 8}, out_channels=32)
model.load_state_dict(torch.load(model_path))
model.eval()

# === 推論スコア計算
with torch.no_grad():
    embeddings = model(data.x_dict, data.edge_index_dict)
    dev_emb = embeddings["dev"][pairs[:, 0]]
    task_emb = embeddings["task"][pairs[:, 1]]
    scores = (dev_emb * task_emb).sum(dim=1)  # dot product

# === 評価：Hit@1, Hit@3, Hit@5（ランキングベース）
def hit_at_k(scores, labels, k=5):
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()

    dev_task = {}
    for (d, t), label, score in zip(pairs.tolist(), labels, scores):
        if d not in dev_task:
            dev_task[d] = []
        dev_task[d].append((t, score, label))

    hits = 0
    total = 0
    for v in dev_task.values():
        ranked = sorted(v, key=lambda x: x[1], reverse=True)[:k]
        if any(lbl == 1 for _, _, lbl in ranked):
            hits += 1
        total += 1

    return hits / total

# === 実行
for k in [1, 3, 5]:
    hit = hit_at_k(scores, labels, k)
    print(f"🎯 Hit@{k}: {hit:.3f}")

# === AUC（全体スコアベース）
try:
    auc = roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy())
    print(f"📈 ROC AUC: {auc:.3f}")
except Exception as e:
    print("⚠️ AUC計算失敗:", e)
