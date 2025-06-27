import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from kazoo.gnn.gnn_model import GNNModel


def evaluate_gnn():
    """学習済みGNNモデルの性能を評価"""

    # パス設定
    root = Path(__file__).resolve().parents[1]
    graph_path = root / "data/graph.pt"
    model_path = root / "data/gnn_model.pt"
    label_path = root / "data/labels.pt"

    # データ読み込み
    print("Loading data...")
    data = torch.load(graph_path, weights_only=False)
    pairs, labels = torch.load(label_path)

    # モデル読み込み
    print("Loading model...")
    model = GNNModel(in_channels_dict={"dev": 8, "task": 9}, out_channels=32)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # 予測実行
    print("Evaluating...")
    with torch.no_grad():
        embeddings = model(data.x_dict, data.edge_index_dict)

        # 有効なペアのみを使用
        valid_mask = (pairs[:, 0] < embeddings["dev"].size(0)) & (
            pairs[:, 1] < embeddings["task"].size(0)
        )
        valid_pairs = pairs[valid_mask]
        valid_labels = labels[valid_mask]

        # 類似度計算
        dev_emb = embeddings["dev"][valid_pairs[:, 0]]
        task_emb = embeddings["task"][valid_pairs[:, 1]]
        scores = F.cosine_similarity(dev_emb, task_emb)

        # 評価指標計算
        y_true = valid_labels.numpy()
        y_scores = scores.numpy()

        roc_auc = roc_auc_score(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)

        print(f"\n📊 評価結果:")
        print(f"  - ROC AUC: {roc_auc:.4f}")
        print(f"  - PR AUC: {pr_auc:.4f}")
        print(f"  - 評価サンプル数: {len(valid_pairs)}")


if __name__ == "__main__":
    evaluate_gnn()
