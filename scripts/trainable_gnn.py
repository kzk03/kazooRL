# ✅ trainable_gnn.py
# GNNを学習するエントリポイント

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F
from torch.optim import Adam

from kazoo.gnn.gnn_model import GNNModel

# === パス設定
root = Path(__file__).resolve().parents[1]
graph_path = root / "data/graph.pt"
label_path = root / "data/labels.pt"

# === データ読み込み
data = torch.load(graph_path, weights_only=False)
pairs, labels = torch.load(label_path)

# === モデル・最適化
model = GNNModel(in_channels_dict={"dev": 8, "task": 8}, out_channels=32)
optimizer = Adam(model.parameters(), lr=1e-3)

# === 学習ループ
for epoch in range(1, 101):
    model.train()
    embeddings = model(data.x_dict, data.edge_index_dict)

    dev_emb = embeddings["dev"][pairs[:, 0]]
    task_emb = embeddings["task"][pairs[:, 1]]

    # 安定化（dot ではなく cosine）
    scores = F.cosine_similarity(dev_emb, task_emb)

    # ラベル安定化
    labels = labels.float()
    if labels.dim() == 2:
        labels = labels.squeeze(1)

    loss = F.binary_cross_entropy_with_logits(scores, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"[Epoch {epoch:03d}] loss: {loss.item():.4f}")

torch.save(model.state_dict(), "data/gnn_model.pt")
print("✅ モデルを保存しました → data/gnn_model.pt")