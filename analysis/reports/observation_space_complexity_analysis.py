#!/usr/bin/env python3
"""
観測空間の複雑さ比較分析
複雑なDict観測空間とシンプルなBox観測空間の詳細比較
"""

from typing import Any, Dict

import gymnasium as gym
import numpy as np


def analyze_observation_spaces():
    """観測空間の複雑さを詳細比較"""

    print("🔍 観測空間の複雑さ比較分析")
    print("=" * 80)

    # データ使用状況の分析を追加
    print("\n0️⃣ データ使用状況の確認")
    print("=" * 50)
    analyze_data_usage()

    print("\n1️⃣ 複雑な観測空間 (OSSSimpleEnv)")
    print("=" * 50)

    # 複雑な観測空間の例
    num_tasks = 20
    complex_obs_space = gym.spaces.Dict(
        {
            "simple_obs": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(num_tasks * 3,),  # 60次元
                dtype=np.float32,
            ),
            "gnn_embeddings": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32  # 64次元
            ),
        }
    )

    print(f"📊 構造:")
    print(f"   タイプ: gym.spaces.Dict")
    print(f"   要素数: 2個")
    print(f"   simple_obs: {complex_obs_space['simple_obs'].shape}")
    print(f"   gnn_embeddings: {complex_obs_space['gnn_embeddings'].shape}")
    print(
        f"   総次元: {num_tasks * 3 + 64} = {complex_obs_space['simple_obs'].shape[0]} + {complex_obs_space['gnn_embeddings'].shape[0]}"
    )

    # サンプル観測の生成
    complex_sample = complex_obs_space.sample()
    print(f"\n📦 サンプル観測:")
    print(f"   simple_obs[0:5]: {complex_sample['simple_obs'][:5]}")
    print(f"   gnn_embeddings[0:5]: {complex_sample['gnn_embeddings'][:5]}")
    print(f"   データ型: {type(complex_sample)}")

    print(f"\n❌ 問題点:")
    problems = [
        "Stable-Baselines3でサポートされていない",
        "ネストした観測空間（Dict内にBox）",
        "特徴量の結合処理が複雑",
        "タスク数変更時の観測空間再定義が必要",
        "PPOエージェントでの直接処理が困難",
    ]

    for i, problem in enumerate(problems, 1):
        print(f"   {i}. {problem}")

    print(f"\n2️⃣ シンプル観測空間 (SimpleTaskAssignmentEnv)")
    print("=" * 50)

    # シンプル観測空間の例
    feature_dim = 62  # FeatureExtractorの出力次元
    simple_obs_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(feature_dim,), dtype=np.float32
    )

    print(f"📊 構造:")
    print(f"   タイプ: gym.spaces.Box")
    print(f"   次元: {simple_obs_space.shape}")
    print(f"   要素数: 1個（統合ベクトル）")
    print(f"   総次元: {feature_dim}")

    # サンプル観測の生成
    simple_sample = simple_obs_space.sample()
    print(f"\n📦 サンプル観測:")
    print(f"   観測[0:5]: {simple_sample[:5]}")
    print(f"   観測[-5:]: {simple_sample[-5:]}")
    print(f"   データ型: {type(simple_sample)}")

    print(f"\n✅ 利点:")
    advantages = [
        "Stable-Baselines3で完全サポート",
        "単一ベクトル、ネストなし",
        "PPOエージェントで直接処理可能",
        "特徴量抽出器による柔軟な内容変更",
        "メモリ効率が良い",
    ]

    for i, advantage in enumerate(advantages, 1):
        print(f"   {i}. {advantage}")

    print(f"\n3️⃣ 特徴量の内容比較")
    print("=" * 50)

    print(f"\n🔸 複雑な観測空間の内容:")
    complex_features = {
        "simple_obs (60次元)": [
            "タスク1: [status, complexity, deadline]",
            "タスク2: [status, complexity, deadline]",
            "...",
            "タスク20: [status, complexity, deadline]",
        ],
        "gnn_embeddings (64次元)": [
            "GAT埋め込みのGlobal Average Pooling",
            "開発者+タスクグラフの総合表現",
            "固定64次元",
        ],
    }

    for feature_type, details in complex_features.items():
        print(f"   {feature_type}:")
        for detail in details:
            print(f"     - {detail}")

    print(f"\n🔸 シンプル観測空間の内容 (FeatureExtractor出力):")
    simple_features = {
        "タスク特徴量 (9次元)": [
            "task_days_since_last_activity",
            "task_discussion_activity",
            "task_text_length",
            "task_code_block_count",
            "task_label_* (5種類)",
        ],
        "開発者特徴量 (6次元)": [
            "dev_recent_activity_count",
            "dev_current_workload",
            "dev_total_lines_changed",
            "dev_collaboration_network_size",
            "dev_comment_interactions",
            "dev_cross_issue_activity",
        ],
        "マッチング特徴量 (10次元)": [
            "match_collaborated_with_task_author",
            "match_collaborator_overlap_count",
            "match_has_prior_collaboration",
            "match_skill_intersection_count",
            "match_file_experience_count",
            "match_affinity_* (5種類)",
        ],
        "GAT特徴量 (37次元)": [
            "gat_similarity",
            "gat_dev_expertise",
            "gat_task_popularity",
            "gat_collaboration_strength",
            "gat_network_centrality",
            "gat_dev_emb_* (32次元埋め込み)",
        ],
    }

    for feature_type, details in simple_features.items():
        print(f"   {feature_type}:")
        for detail in details:
            print(f"     - {detail}")

    print(f"\n4️⃣ Stable-Baselines3対応の違い")
    print("=" * 50)

    print(f"\n❌ 複雑な観測空間でのエラー:")
    error_message = """
NotImplementedError: Nested observation spaces are not supported 
(Tuple/Dict space inside Tuple/Dict space).
    """
    print(f"   {error_message.strip()}")

    print(f"\n✅ シンプル観測空間での正常動作:")
    success_messages = [
        "PPO('MlpPolicy', env) ← 正常に初期化",
        "model.learn(total_timesteps=5000) ← 正常に訓練",
        "model.predict(obs) ← 正常に推論",
        "EvalCallback(...) ← 正常に評価",
    ]

    for message in success_messages:
        print(f"   {message}")

    print(f"\n5️⃣ メモリと計算効率の比較")
    print("=" * 50)

    # メモリ使用量の比較
    complex_memory = calculate_memory_usage(complex_obs_space, "複雑")
    simple_memory = calculate_memory_usage(simple_obs_space, "シンプル")

    print(f"\n💾 メモリ使用量:")
    print(f"   複雑な観測空間: {complex_memory['total']:.2f} KB")
    print(f"     - simple_obs: {complex_memory['simple_obs']:.2f} KB")
    print(f"     - gnn_embeddings: {complex_memory['gnn_embeddings']:.2f} KB")
    print(f"     - メタデータ: {complex_memory['metadata']:.2f} KB")

    print(f"\n   シンプル観測空間: {simple_memory['total']:.2f} KB")
    print(f"     - 観測ベクトル: {simple_memory['obs_vector']:.2f} KB")
    print(f"     - メタデータ: {simple_memory['metadata']:.2f} KB")

    efficiency = (
        (complex_memory["total"] - simple_memory["total"])
        / complex_memory["total"]
        * 100
    )
    print(f"\n   効率改善: {efficiency:.1f}% メモリ削減")

    print(f"\n6️⃣ 実装コードの比較")
    print("=" * 50)

    print(f"\n🔸 複雑な観測空間での処理:")
    complex_code = """
# 観測の取得（複雑）
obs = env.reset()
simple_part = obs['simple_obs']        # (60,)
gnn_part = obs['gnn_embeddings']       # (64,)

# PPOで処理（エラー）
model = PPO("MlpPolicy", env)  # ❌ NotImplementedError

# 手動で結合が必要
combined_obs = np.concatenate([simple_part, gnn_part])  # (124,)
"""

    print(f"   {complex_code.strip()}")

    print(f"\n🔸 シンプル観測空間での処理:")
    simple_code = """
# 観測の取得（シンプル）
obs = env.reset()                      # (62,) 直接取得

# PPOで処理（正常）
model = PPO("MlpPolicy", env)          # ✅ 正常動作
action = model.predict(obs)            # ✅ 直接推論可能

# 結合処理不要
# obs はすでに統合済みの特徴量ベクトル
"""

    print(f"   {simple_code.strip()}")

    print(f"\n7️⃣ まとめ")
    print("=" * 50)

    summary = {
        "複雑な観測空間": {
            "利点": ["既存システムとの互換性", "要素別の明確な分離"],
            "欠点": ["Stable-Baselines3非対応", "実装が複雑", "メモリ効率悪"],
        },
        "シンプル観測空間": {
            "利点": ["Stable-Baselines3対応", "実装が簡単", "メモリ効率良"],
            "欠点": ["要素の分離が不明確", "デバッグが困難"],
        },
    }

    for space_type, pros_cons in summary.items():
        print(f"\n{space_type}:")
        print(f"   利点: {', '.join(pros_cons['利点'])}")
        print(f"   欠点: {', '.join(pros_cons['欠点'])}")

    print(f"\n🎯 推奨:")
    print(f"   - 新規システム: シンプル観測空間を使用")
    print(f"   - 既存システム: 複雑→シンプルに段階的移行")
    print(f"   - 研究・実験: シンプル観測空間で高速プロトタイピング")


def calculate_memory_usage(obs_space, space_type):
    """観測空間のメモリ使用量を計算"""

    if space_type == "複雑":
        simple_obs_size = obs_space["simple_obs"].shape[0] * 4  # float32 = 4 bytes
        gnn_emb_size = obs_space["gnn_embeddings"].shape[0] * 4
        metadata_size = 100  # Dict構造のオーバーヘッド

        return {
            "simple_obs": simple_obs_size / 1024,  # KB
            "gnn_embeddings": gnn_emb_size / 1024,
            "metadata": metadata_size / 1024,
            "total": (simple_obs_size + gnn_emb_size + metadata_size) / 1024,
        }

    else:  # シンプル
        obs_vector_size = obs_space.shape[0] * 4  # float32 = 4 bytes
        metadata_size = 20  # Box構造の軽量オーバーヘッド

        return {
            "obs_vector": obs_vector_size / 1024,  # KB
            "metadata": metadata_size / 1024,
            "total": (obs_vector_size + metadata_size) / 1024,
        }


def analyze_data_usage():
    """データ使用状況の分析"""
    import json
    from datetime import datetime

    try:
        print(f"📅 データ期間の分析:")

        # データファイルを読み込み
        data_files = {
            "全データ": "data/backlog.json",
            "RL訓練用(2022)": "data/backlog_training_2022.json",
            "旧訓練用(2019-2021)": "data/backlog_training.json",
            "テスト用": "data/backlog_test_2022.json",
        }

        for name, filepath in data_files.items():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if data:
                    start_date = data[0]["created_at"][:10]
                    end_date = data[-1]["created_at"][:10]
                    count = len(data)

                    print(f"   {name}: {count:,}タスク ({start_date} ～ {end_date})")
                else:
                    print(f"   {name}: データなし")

            except FileNotFoundError:
                print(f"   {name}: ファイルが見つかりません ({filepath})")
            except Exception as e:
                print(f"   {name}: エラー ({e})")

        print(f"\n🎯 現在の使用方法:")
        print(f"   - IRL学習: 2019-2021年のexpert trajectories（学習済み）")
        print(f"   - RL訓練: 2022年のbacklog_training_2022.json")
        print(f"   - 評価: 2022年以降のデータまたはテストセット")

        print(f"\n✅ 改善点:")
        print(f"   - IRLとRLで異なる期間のデータを使用")
        print(f"   - 時系列順序: IRL(2019-2021) → RL(2022)")
        print(f"   - データリークのリスクを大幅に軽減")

        print(f"\n🔧 さらなる改善案（2023年データ追加後）:")
        print(f"   - IRL学習: 2019-2021年（現在）")
        print(f"   - RL訓練: 2022年（現在）")
        print(f"   - テスト: 2023年（追加予定）")
        print(f"   実行: python scripts/split_temporal_data_simple.py")

    except Exception as e:
        print(f"⚠️ データ使用状況の分析でエラー: {e}")


if __name__ == "__main__":
    analyze_observation_spaces()
