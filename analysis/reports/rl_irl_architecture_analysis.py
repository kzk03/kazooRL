#!/usr/bin/env python3
"""
強化学習と逆強化学習のアーキテクチャ分析

エージェント数、状態、行動、報酬の整合性と最適化について詳細分析
"""

from datetime import datetime

import numpy as np
import pandas as pd


def analyze_rl_irl_architecture():
    """強化学習と逆強化学習のアーキテクチャ分析"""

    print("=" * 80)
    print("🔍 強化学習・逆強化学習アーキテクチャ分析")
    print("=" * 80)

    print("\n1️⃣ 現在の実装構造")
    print("=" * 50)

    architecture_info = {
        "環境": "OSSSimpleEnv (Multi-Agent)",
        "エージェント数": "動的設定可能 (num_developers)",
        "状態空間": "Dict観測 (simple_obs + gnn_embeddings)",
        "行動空間": "Discrete(タスク数+1) 各エージェント",
        "報酬計算": "IRL学習重み + デフォルト報酬",
        "PPOエージェント": "独立学習 (各開発者別)",
        "GAT特徴量": "32次元埋め込み + 統計特徴量",
    }

    for key, value in architecture_info.items():
        print(f"   {key}: {value}")

    print(f"\n2️⃣ エージェント数の整合性問題")
    print("=" * 50)

    consistency_issues = [
        {
            "問題": "IRL訓練データとRL実行時の開発者数不一致",
            "詳細": "expert_trajectories.pklの開発者数 ≠ num_developers設定",
            "影響": "学習済み重みが一部の開発者にしか適用されない",
            "重要度": "🔴 高",
        },
        {
            "問題": "GAT特徴量の次元不整合",
            "詳細": "32次元埋め込み + 64次元プール = 非効率な結合",
            "影響": "特徴量表現力の低下、学習効率の悪化",
            "重要度": "🟡 中",
        },
        {
            "問題": "観測空間の固定サイズ制約",
            "詳細": "タスク数 * 3の固定長 + 64次元GNN",
            "影響": "スケーラビリティの制限",
            "重要度": "🟡 中",
        },
        {
            "問題": "報酬重みの次元一致",
            "詳細": "IRL特徴量次元 ≠ RL観測次元",
            "影響": "報酬計算でのエラーやデフォルト報酬への依存",
            "重要度": "🔴 高",
        },
    ]

    for i, issue in enumerate(consistency_issues, 1):
        print(f"\n{i}. {issue['問題']} {issue['重要度']}")
        print(f"   詳細: {issue['詳細']}")
        print(f"   影響: {issue['影響']}")

    print(f"\n3️⃣ 状態・行動・報酬の詳細")
    print("=" * 50)

    print(f"\n📊 状態空間 (Observation Space)")
    print("-" * 30)
    state_components = [
        {
            "要素": "simple_obs",
            "形状": "(タスク数 * 3,)",
            "内容": "[status, complexity, deadline] × タスク数",
            "例": "20タスク → (60,)",
        },
        {
            "要素": "gnn_embeddings",
            "形状": "(64,)",
            "内容": "GAT埋め込みのGlobal Average Pooling",
            "例": "開発者+タスク埋め込み → (64,)",
        },
        {
            "要素": "total_obs_dim",
            "形状": "(124,)",
            "内容": "60 + 64 = 124次元",
            "例": "PPOエージェントの入力次元",
        },
    ]

    for comp in state_components:
        print(f"   {comp['要素']}: {comp['形状']}")
        print(f"     内容: {comp['内容']}")
        print(f"     例: {comp['例']}")

    print(f"\n🎯 行動空間 (Action Space)")
    print("-" * 30)
    action_info = [
        {
            "空間": "Discrete(タスク数 + 1)",
            "意味": "選択するタスクのインデックス + NO-OP",
            "例": "20タスク → Discrete(21)",
            "制約": "同じタスクを複数エージェントが選択可能",
        },
        {
            "空間": "Multi-Agent",
            "意味": "各開発者が独立して行動選択",
            "例": "20人 → 20個の独立したDiscrete(21)",
            "制約": "同期実行、全エージェント同時決定",
        },
    ]

    for action in action_info:
        print(f"   {action['空間']}: {action['意味']}")
        print(f"     例: {action['例']}")
        print(f"     制約: {action['制約']}")

    print(f"\n💰 報酬構造 (Reward Structure)")
    print("-" * 30)
    reward_components = [
        {
            "タイプ": "IRL学習報酬",
            "計算": "np.dot(reward_weights, features)",
            "特徴量": "FeatureExtractor出力 (可変次元)",
            "適用": "reward_weights_path指定時",
        },
        {
            "タイプ": "デフォルト報酬",
            "計算": "完了=1.0, その他=0.0",
            "特徴量": "なし",
            "適用": "IRL重みなし or エラー時",
        },
        {
            "タイプ": "GNNインタラクション記録",
            "計算": "報酬値そのまま記録",
            "特徴量": "オンライン学習用バッファ",
            "適用": "GAT更新時",
        },
    ]

    for reward in reward_components:
        print(f"   {reward['タイプ']}: {reward['計算']}")
        print(f"     特徴量: {reward['特徴量']}")
        print(f"     適用: {reward['適用']}")

    print(f"\n4️⃣ 推奨されるアーキテクチャ改善")
    print("=" * 50)

    improvements = [
        {
            "改善項目": "エージェント数の統一",
            "現状問題": "IRL訓練とRL実行で開発者数が異なる",
            "推奨解決": "expert_trajectories生成時にnum_developers統一",
            "実装": "create_expert_trajectories.pyでnum_developers指定",
            "優先度": "🔴 最高",
        },
        {
            "改善項目": "観測空間の最適化",
            "現状問題": "simple_obs + gnn_embeddingsの非効率結合",
            "推奨解決": "GAT特徴量を主とし、統計特徴量を補完",
            "実装": "GNNFeatureExtractor出力を直接使用",
            "優先度": "🟡 中",
        },
        {
            "改善項目": "報酬計算の安定化",
            "現状問題": "IRL特徴量とRL観測の次元不一致",
            "推奨解決": "特徴量抽出器の統一、エラーハンドリング強化",
            "実装": "FeatureExtractor.get_features()の改良",
            "優先度": "🔴 高",
        },
        {
            "改善項目": "スケーラビリティ向上",
            "現状問題": "タスク数固定の観測空間",
            "推奨解決": "動的サイズ対応、attention機構導入",
            "実装": "Transformer-based観測エンコーダ",
            "優先度": "🟢 低",
        },
    ]

    for i, improvement in enumerate(improvements, 1):
        print(f"\n{i}. {improvement['改善項目']} {improvement['優先度']}")
        print(f"   現状問題: {improvement['現状問題']}")
        print(f"   推奨解決: {improvement['推奨解決']}")
        print(f"   実装方法: {improvement['実装']}")

    print(f"\n5️⃣ 開発者数別の推奨設定")
    print("=" * 50)

    developer_configs = {
        20: {
            "irl_experts": "20人のexpert_trajectories生成",
            "rl_agents": "20人のPPOエージェント",
            "observation_dim": "124次元 (60 + 64)",
            "action_space": "Discrete(21) × 20",
            "memory_usage": "低 (~2GB)",
            "training_time": "短 (~1時間)",
        },
        50: {
            "irl_experts": "50人のexpert_trajectories生成",
            "rl_agents": "50人のPPOエージェント",
            "observation_dim": "124次元 (60 + 64)",
            "action_space": "Discrete(21) × 50",
            "memory_usage": "中 (~4GB)",
            "training_time": "中 (~2時間)",
        },
        200: {
            "irl_experts": "200人のexpert_trajectories生成",
            "rl_agents": "200人のPPOエージェント",
            "observation_dim": "124次元 (60 + 64)",
            "action_space": "Discrete(21) × 200",
            "memory_usage": "高 (~8GB)",
            "training_time": "長 (~4時間)",
        },
        1000: {
            "irl_experts": "1000人のexpert_trajectories生成",
            "rl_agents": "1000人のPPOエージェント",
            "observation_dim": "124次元 (60 + 64)",
            "action_space": "Discrete(21) × 1000",
            "memory_usage": "非常に高 (~32GB)",
            "training_time": "非常に長 (~12時間)",
        },
    }

    for dev_count, config in developer_configs.items():
        print(f"\n👥 {dev_count:,}人設定:")
        for key, value in config.items():
            print(f"   {key}: {value}")

    print(f"\n6️⃣ 緊急対応が必要な問題")
    print("=" * 50)

    urgent_issues = [
        {
            "問題": "expert_trajectories.pklの開発者数確認",
            "確認方法": "pickle.load()して軌跡内の開発者数を数える",
            "対応": "num_developersと一致しない場合は再生成",
            "コマンド": "python tools/analysis_and_debug/debug_review_match.py",
        },
        {
            "問題": "reward_weightsの次元確認",
            "確認方法": "learned_weights_training.npyの形状確認",
            "対応": "特徴量次元と一致しない場合はIRL再訓練",
            "コマンド": "python training/irl/train_irl.py",
        },
        {
            "問題": "GAT特徴量の整合性確認",
            "確認方法": "GNNFeatureExtractorの出力次元確認",
            "対応": "32次元埋め込みと統計特徴量の適切な結合",
            "コマンド": "python analysis/reports/gat_features_detailed_analysis.py",
        },
    ]

    for i, issue in enumerate(urgent_issues, 1):
        print(f"\n{i}. {issue['問題']}")
        print(f"   確認方法: {issue['確認方法']}")
        print(f"   対応: {issue['対応']}")
        print(f"   コマンド: {issue['コマンド']}")

    print(f"\n" + "=" * 80)
    print("🎯 次のアクション推奨")
    print("=" * 80)

    next_actions = [
        "1. expert_trajectories.pklの開発者数を確認",
        "2. num_developers=20に合わせてexpert_trajectories再生成",
        "3. IRL重みの次元とRL特徴量次元の整合性確認",
        "4. 20人設定でのRL実行テスト",
        "5. 問題ない場合は200人→1000人にスケールアップ",
    ]

    for action in next_actions:
        print(f"   {action}")

    # CSV出力
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 問題分析結果をDataFrame化
    issues_df = pd.DataFrame(consistency_issues)
    issues_csv = f"outputs/rl_irl_issues_analysis_{timestamp}.csv"
    issues_df.to_csv(issues_csv, index=False, encoding="utf-8")

    # 設定推奨をDataFrame化
    config_df = pd.DataFrame.from_dict(developer_configs, orient="index")
    config_csv = f"outputs/rl_irl_config_recommendations_{timestamp}.csv"
    config_df.to_csv(config_csv, index=False, encoding="utf-8")

    print(f"\n💾 分析結果をCSVに保存:")
    print(f"   問題分析: {issues_csv}")
    print(f"   設定推奨: {config_csv}")

    return issues_df, config_df


if __name__ == "__main__":
    analyze_rl_irl_architecture()
