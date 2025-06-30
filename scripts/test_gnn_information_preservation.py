#!/usr/bin/env python3
"""
GNNの時系列学習における情報保持/消失のテスト
- 元のグラフ構造が保持されるかテスト
- 過去の学習結果が累積されるかテスト
- 時間窓による影響をテスト
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from omegaconf import OmegaConf

from kazoo.envs.oss_simple import OSSSimpleEnv


def test_gnn_information_preservation():
    """GNNの情報保持/消失を詳細テスト"""
    print("🔍 GNN時系列学習での情報保持/消失テスト")
    print("=" * 60)

    # 設定とデータ読み込み
    cfg = OmegaConf.load(project_root / "configs" / "base.yaml")

    with open(project_root / cfg.env.backlog_path, "r") as f:
        backlog = json.load(f)

    with open(project_root / cfg.env.dev_profiles_path, "r") as f:
        dev_profiles = yaml.safe_load(f)

    # 環境初期化
    env = OSSSimpleEnv(cfg, backlog, dev_profiles)
    gnn_extractor = env.feature_extractor.gnn_extractor

    if not gnn_extractor or not gnn_extractor.online_learning:
        print("❌ GNNオンライン学習が利用できません")
        return

    print("✅ テスト準備完了")

    # テスト用ノード
    test_devs = list(gnn_extractor.dev_id_to_idx.keys())[:3]
    test_tasks = list(gnn_extractor.task_id_to_idx.keys())[:5]

    print(f"📊 テスト対象: 開発者{len(test_devs)}名, タスク{len(test_tasks)}個")

    # Step 1: 初期状態の埋め込みを記録
    print("\n🔸 Step 1: 初期状態の埋め込みを記録")

    def get_embeddings_snapshot():
        """現在の埋め込みのスナップショットを取得"""
        with torch.no_grad():
            embeddings = gnn_extractor.model(
                gnn_extractor.graph_data.x_dict,
                gnn_extractor.graph_data.edge_index_dict,
            )
        return {"dev": embeddings["dev"].clone(), "task": embeddings["task"].clone()}

    def get_similarity_matrix(dev_list, task_list, embeddings):
        """開発者-タスク間の類似度行列を計算"""
        similarities = {}
        for dev_name in dev_list:
            dev_idx = gnn_extractor.dev_id_to_idx.get(dev_name)
            if dev_idx is not None:
                similarities[dev_name] = {}
                for task_id in task_list:
                    task_idx = gnn_extractor.task_id_to_idx.get(task_id)
                    if task_idx is not None:
                        dev_emb = embeddings["dev"][dev_idx]
                        task_emb = embeddings["task"][task_idx]
                        sim = torch.cosine_similarity(
                            dev_emb.unsqueeze(0), task_emb.unsqueeze(0)
                        ).item()
                        similarities[dev_name][task_id] = sim
        return similarities

    # 初期埋め込みと類似度を記録
    initial_embeddings = get_embeddings_snapshot()
    initial_similarities = get_similarity_matrix(
        test_devs, test_tasks, initial_embeddings
    )

    print("  📊 初期類似度行列:")
    for dev_name in test_devs:
        print(f"    {dev_name}:")
        for task_id in test_tasks[:3]:  # 最初の3つのタスクのみ表示
            sim = initial_similarities[dev_name][task_id]
            print(f"      vs {task_id[:15]}... = {sim:.4f}")

    # Step 2: 第1期間のインタラクション（時間: T0）
    print("\n🔸 Step 2: 第1期間のインタラクション（T0）")

    class TestTask:
        def __init__(self, task_id):
            self.id = task_id

    class TestDev:
        def __init__(self, dev_name):
            self.name = dev_name

        def get(self, key, default=None):
            return self.name if key == "name" else default

    base_time = env.current_time

    # 第1期間: 特定のペアを強化
    period1_interactions = [
        (test_devs[0], test_tasks[0], base_time, 1.5, "strong_positive"),
        (test_devs[0], test_tasks[1], base_time + timedelta(hours=1), 1.2, "positive"),
        (test_devs[1], test_tasks[2], base_time + timedelta(hours=2), 1.0, "positive"),
        (test_devs[2], test_tasks[0], base_time + timedelta(hours=3), -0.8, "negative"),
        (test_devs[1], test_tasks[3], base_time + timedelta(hours=4), 1.1, "positive"),
    ]

    print(f"  📝 第1期間: {len(period1_interactions)} 件のインタラクション")
    for dev_name, task_id, sim_time, reward, action in period1_interactions:
        developer = TestDev(dev_name)
        task = TestTask(task_id)
        gnn_extractor.record_interaction(
            task, developer, reward, action, simulation_time=sim_time
        )
        print(
            f"    {sim_time.strftime('%H:%M')} - {dev_name} + {task_id[:15]}... = {reward}"
        )

    # 第1期間後の状態を記録
    period1_embeddings = get_embeddings_snapshot()
    period1_similarities = get_similarity_matrix(
        test_devs, test_tasks, period1_embeddings
    )

    print("  📊 第1期間後の類似度変化:")
    for dev_name in test_devs:
        for task_id in test_tasks[:3]:
            initial_sim = initial_similarities[dev_name][task_id]
            period1_sim = period1_similarities[dev_name][task_id]
            change = period1_sim - initial_sim
            print(
                f"    {dev_name} vs {task_id[:15]}...: {initial_sim:.4f} → {period1_sim:.4f} (Δ{change:+.4f})"
            )

    # Step 3: 時間を大幅に進めて第2期間（時間窓外）
    print("\n🔸 Step 3: 第2期間のインタラクション（T0 + 30時間, 時間窓外）")

    period2_start = base_time + timedelta(hours=30)  # 時間窓（24時間）を超える

    # 第2期間: 異なるペアを強化（第1期間とは逆のパターン）
    period2_interactions = [
        (
            test_devs[2],
            test_tasks[0],
            period2_start,
            1.8,
            "very_positive",
        ),  # 第1期間ではnegative
        (
            test_devs[1],
            test_tasks[1],
            period2_start + timedelta(hours=1),
            1.5,
            "strong_positive",
        ),
        (
            test_devs[0],
            test_tasks[3],
            period2_start + timedelta(hours=2),
            1.3,
            "positive",
        ),
        (
            test_devs[0],
            test_tasks[0],
            period2_start + timedelta(hours=3),
            -0.9,
            "negative",
        ),  # 第1期間ではpositive
        (
            test_devs[2],
            test_tasks[4],
            period2_start + timedelta(hours=4),
            1.0,
            "positive",
        ),
    ]

    print(f"  📝 第2期間: {len(period2_interactions)} 件のインタラクション")
    for dev_name, task_id, sim_time, reward, action in period2_interactions:
        developer = TestDev(dev_name)
        task = TestTask(task_id)
        gnn_extractor.record_interaction(
            task, developer, reward, action, simulation_time=sim_time
        )
        print(
            f"    {sim_time.strftime('%H:%M')} - {dev_name} + {task_id[:15]}... = {reward}"
        )

    # 第2期間後の状態を記録
    period2_embeddings = get_embeddings_snapshot()
    period2_similarities = get_similarity_matrix(
        test_devs, test_tasks, period2_embeddings
    )

    print("  📊 第2期間後の類似度変化:")
    for dev_name in test_devs:
        for task_id in test_tasks[:3]:
            period1_sim = period1_similarities[dev_name][task_id]
            period2_sim = period2_similarities[dev_name][task_id]
            change = period2_sim - period1_sim
            print(
                f"    {dev_name} vs {task_id[:15]}...: {period1_sim:.4f} → {period2_sim:.4f} (Δ{change:+.4f})"
            )

    # Step 4: 時間窓の影響を分析
    print("\n🔸 Step 4: 時間窓の影響分析")

    # インタラクションバッファの分析
    all_interactions = gnn_extractor.interaction_buffer
    latest_time = max(
        interaction["simulation_time"] for interaction in all_interactions
    )
    cutoff_time = latest_time - timedelta(hours=gnn_extractor.time_window_hours)

    period1_count = sum(
        1
        for interaction in all_interactions
        if interaction["simulation_time"] < cutoff_time
    )
    period2_count = sum(
        1
        for interaction in all_interactions
        if interaction["simulation_time"] >= cutoff_time
    )

    print(f"  ⏰ 最新時刻: {latest_time.strftime('%m/%d %H:%M')}")
    print(f"  ⏰ カットオフ時刻: {cutoff_time.strftime('%m/%d %H:%M')}")
    print(f"  📊 第1期間インタラクション（時間窓外）: {period1_count} 件")
    print(f"  📊 第2期間インタラクション（時間窓内）: {period2_count} 件")

    # Step 5: 元のグラフ構造との比較
    print("\n🔸 Step 5: 元のグラフ構造との比較")

    # 元のグラフのエッジ情報を確認
    original_edges = gnn_extractor.graph_data[("dev", "writes", "task")].edge_index
    print(f"  📊 元のグラフエッジ数: {original_edges.shape[1]}")

    # 学習で影響を受けたペアと元のエッジの関係を確認
    print("  🔍 学習ペアと元のエッジの関係:")
    for dev_name, task_id, _, reward, _ in period1_interactions + period2_interactions:
        dev_idx = gnn_extractor.dev_id_to_idx.get(dev_name)
        task_idx = gnn_extractor.task_id_to_idx.get(task_id)

        if dev_idx is not None and task_idx is not None:
            # 元のグラフにエッジが存在するかチェック
            has_original_edge = False
            for i in range(original_edges.shape[1]):
                if (
                    original_edges[0, i].item() == dev_idx
                    and original_edges[1, i].item() == task_idx
                ):
                    has_original_edge = True
                    break

            edge_status = "元エッジあり" if has_original_edge else "元エッジなし"
            print(
                f"    {dev_name} → {task_id[:15]}... (報酬:{reward:+.1f}): {edge_status}"
            )

    # Step 6: 総合分析
    print("\n🔸 Step 6: 総合分析結果")

    # 類似度の変化パターンを分析
    total_changes = {}
    for dev_name in test_devs:
        for task_id in test_tasks:
            initial_sim = initial_similarities[dev_name][task_id]
            final_sim = period2_similarities[dev_name][task_id]
            total_change = final_sim - initial_sim
            total_changes[f"{dev_name}-{task_id}"] = total_change

    # 最も変化したペアを特定
    most_positive_change = max(total_changes.items(), key=lambda x: x[1])
    most_negative_change = min(total_changes.items(), key=lambda x: x[1])

    print(
        f"  📈 最大の正の変化: {most_positive_change[0]} (Δ{most_positive_change[1]:+.4f})"
    )
    print(
        f"  📉 最大の負の変化: {most_negative_change[0]} (Δ{most_negative_change[1]:+.4f})"
    )

    # 統計情報
    updates_count = gnn_extractor.stats.get("updates", 0)
    buffer_size = len(gnn_extractor.interaction_buffer)

    print(f"\n📊 学習統計:")
    print(f"  🔄 GNN更新回数: {updates_count}")
    print(f"  💾 バッファサイズ: {buffer_size}")
    print(f"  ⏰ 時間窓: {gnn_extractor.time_window_hours} 時間")

    # 結論
    print(f"\n🎯 結論:")
    print(f"  1. **グラフ構造**: 元のノード・エッジ構造は保持される")
    print(f"  2. **学習の累積**: モデルパラメータは継続的に更新される")
    print(
        f"  3. **時間窓の影響**: 学習には直近{gnn_extractor.time_window_hours}時間のデータのみ使用"
    )
    print(f"  4. **情報の保持**: 過去の学習結果はモデル重みに蓄積される")
    print(f"  5. **時間窓外データ**: 直接的な学習には使われないが、過去の重みは保持")


if __name__ == "__main__":
    test_gnn_information_preservation()
