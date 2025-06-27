import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import json
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

from kazoo.gnn.gnn_model import GNNModel


def analyze_recommendations_with_metadata():
    """推薦結果とメタデータを組み合わせた詳細分析"""

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

    # ノードIDを取得
    if hasattr(data["dev"], "node_id") and isinstance(data["dev"].node_id, list):
        dev_ids = data["dev"].node_id
    else:
        dev_ids = list(range(embeddings["dev"].size(0)))

    if hasattr(data["task"], "node_id") and isinstance(data["task"].node_id, list):
        task_ids = data["task"].node_id
    else:
        task_ids = list(range(embeddings["task"].size(0)))

    print("🔍 推薦システムの詳細分析")
    print(f"開発者数: {len(dev_ids)}, タスク数: {len(task_ids)}")

    # 1. 開発者ID別の推薦パターン分析
    print("\n📋 開発者ID別推薦パターン")

    recommendation_patterns = defaultdict(list)
    for dev_idx in range(min(20, embeddings["dev"].size(0))):
        dev_vec = embeddings["dev"][dev_idx : dev_idx + 1]
        similarities = F.cosine_similarity(dev_vec, embeddings["task"])
        top_5 = torch.topk(similarities, k=5)

        dev_id = dev_ids[dev_idx] if dev_idx < len(dev_ids) else f"dev_{dev_idx}"
        pattern = []

        for sim, task_idx in zip(top_5.values, top_5.indices):
            task_id = (
                task_ids[task_idx] if task_idx < len(task_ids) else f"task_{task_idx}"
            )
            pattern.append((task_id, sim.item()))

        recommendation_patterns[str(dev_id)] = pattern

    # パターンの類似性を分析
    print("推薦パターンの多様性:")
    unique_patterns = {}
    for dev_id, pattern in recommendation_patterns.items():
        # タスクIDだけを取得してパターンを作成
        task_pattern = tuple([task_id for task_id, _ in pattern])
        if task_pattern not in unique_patterns:
            unique_patterns[task_pattern] = []
        unique_patterns[task_pattern].append(dev_id)

    print(f"  ユニークなパターン数: {len(unique_patterns)}")
    for i, (pattern, devs) in enumerate(unique_patterns.items()):
        if i < 5:  # 上位5パターンを表示
            print(f"  パターン{i+1}: {len(devs)}名の開発者")
            print(f"    タスク: {pattern[:3]}...")  # 最初の3タスクを表示
            print(f"    開発者: {devs[:5]}")  # 最初の5名を表示

    # 2. タスクID分析
    print("\n🎯 推薦されやすいタスクの分析")

    # 全開発者に対する各タスクの推薦頻度を計算
    task_recommendation_count = Counter()

    for dev_idx in range(
        min(50, embeddings["dev"].size(0))
    ):  # 50名の開発者でサンプリング
        dev_vec = embeddings["dev"][dev_idx : dev_idx + 1]
        similarities = F.cosine_similarity(dev_vec, embeddings["task"])
        top_10 = torch.topk(similarities, k=10).indices  # トップ10を考慮

        for task_idx in top_10:
            task_id = (
                task_ids[task_idx] if task_idx < len(task_ids) else f"task_{task_idx}"
            )
            task_recommendation_count[task_id] += 1

    print("最も推薦されやすいタスク（トップ10）:")
    for i, (task_id, count) in enumerate(task_recommendation_count.most_common(10)):
        print(f"  {i+1}. {task_id}: {count}回推薦")

    # 3. 類似度値の分布詳細分析
    print("\n📊 類似度値の詳細分布")

    all_similarities = []
    for dev_idx in range(min(20, embeddings["dev"].size(0))):
        dev_vec = embeddings["dev"][dev_idx : dev_idx + 1]
        similarities = F.cosine_similarity(dev_vec, embeddings["task"])
        all_similarities.extend(similarities.tolist())

    import numpy as np

    all_similarities = np.array(all_similarities)

    print(f"類似度統計:")
    print(f"  サンプル数: {len(all_similarities)}")
    print(f"  平均値: {np.mean(all_similarities):.4f}")
    print(f"  中央値: {np.median(all_similarities):.4f}")
    print(f"  標準偏差: {np.std(all_similarities):.4f}")
    print(f"  最小値: {np.min(all_similarities):.4f}")
    print(f"  最大値: {np.max(all_similarities):.4f}")

    # パーセンタイル分析
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("パーセンタイル分析:")
    for p in percentiles:
        value = np.percentile(all_similarities, p)
        print(f"  {p}%tile: {value:.4f}")

    # 4. 実際のエッジ関係との比較
    print("\n🔗 実際のエッジ関係との比較")

    # 実際のエッジを取得
    writes_edges = data[("dev", "writes", "task")].edge_index
    actual_relationships = set()

    for i in range(writes_edges.size(1)):
        dev_idx = writes_edges[0, i].item()
        task_idx = writes_edges[1, i].item()
        if dev_idx < len(dev_ids) and task_idx < len(task_ids):
            dev_id = dev_ids[dev_idx]
            task_id = task_ids[task_idx]
            actual_relationships.add((dev_id, task_id))

    print(f"実際の関係数: {len(actual_relationships)}")

    # 推薦結果と実際の関係の一致度を確認
    match_count = 0
    total_recommendations = 0

    for dev_id, recommendations in list(recommendation_patterns.items())[:10]:
        for task_id, _ in recommendations:
            total_recommendations += 1
            if (dev_id, task_id) in actual_relationships:
                match_count += 1

    if total_recommendations > 0:
        match_rate = match_count / total_recommendations
        print(
            f"推薦と実際の関係の一致率: {match_rate:.2%} ({match_count}/{total_recommendations})"
        )

    return recommendation_patterns, task_recommendation_count


if __name__ == "__main__":
    analyze_recommendations_with_metadata()
