import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.serialization import add_safe_globals
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import BaseStorage

from kazoo.gnn.gnn_model import GNNModel


def main():
    # === パス設定 ===
    root = Path(__file__).resolve().parents[1]
    graph_path = root / "data/graph.pt"
    emb_output_path = root / "data/dev_embeddings.pt"

    # === PyTorch 2.6+ セキュリティ対策 ===
    add_safe_globals([HeteroData, BaseStorage])

    # === グラフ読み込み ===
    data = torch.load(graph_path, weights_only=False)
    print("✅ グラフ読み込み成功")

    # === GNNモデル構築 ===
    model = GNNModel(in_channels=5, out_channels=32)
    model.eval()

    # === 推論実行 ===
    with torch.no_grad():
        embeddings = model(data.x_dict, data.edge_index_dict)

    # === 埋め込み保存 ===
    torch.save(embeddings["dev"], emb_output_path)
    print(f"✅ dev埋め込み保存 → {emb_output_path}")

    # === 確認出力 ===
    print("📐 dev 埋め込みサイズ:", embeddings["dev"].shape)
    print("📐 task 埋め込みサイズ:", embeddings["task"].shape)

    # === 推薦スコアの例 ===
    dev_idx = 0
    task_idx = 7
    score = torch.dot(embeddings["dev"][dev_idx], embeddings["task"][task_idx])
    print(f"💡 dev_{dev_idx} vs task_{task_idx} スコア: {score:.4f}")

    # === タスク推薦ランキング（dev_0対象） ===
    scores = torch.matmul(embeddings["task"], embeddings["dev"][dev_idx])
    topk = torch.topk(scores, k=5)
    print(f"\n📊 dev_{dev_idx} におすすめのタスクTOP5:")
    for i, idx in enumerate(topk.indices):
        print(f"  #{i+1}: task_{idx.item()}（スコア: {topk.values[i]:.4f})")

if __name__ == "__main__":
    main()
