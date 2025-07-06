#!/usr/bin/env python3
"""
Kazoo プロジェクト構造整理計画
現在の構造を分析し、機能別に整理された新しい階層構造を提案
"""

import shutil
from datetime import datetime
from pathlib import Path


def analyze_current_structure():
    """現在のプロジェクト構造を分析"""
    print("📁 現在のプロジェクト構造分析")
    print("=" * 60)

    # 現在の分析・実行スクリプトをカテゴリ分け
    root_scripts = {
        "パイプライン": ["run_full_training_from_scratch.py"],
        "分析・レポート": [
            "analyze_collaboration.py",
            "analyze_gat_features.py",
            "analyze_irl_collaboration.py",
            "analyze_irl_completion.py",
            "analyze_irl_feature_weights.py",
            "analyze_training_results.py",
            "generate_summary_report.py",
            "simple_irl_analysis.py",
        ],
        "テスト・デバッグ": ["test_feature_dimensions.py"],
    }

    scripts_dir = {
        "GAT関連": [
            "scripts/train_collaborative_gat.py",
            "scripts/train_gnn.py",
            "scripts/retrain_gnn_with_recent_data.py",
            "scripts/plot_gnn_graph.py",
        ],
        "IRL関連": ["scripts/train_irl.py"],
        "RL関連": ["scripts/train_oss.py"],
        "評価": ["scripts/evaluate_2022_test.py"],
        "パイプライン": [
            "scripts/full_training_pipeline.py",
            "scripts/run_complete_pipeline.py",
        ],
    }

    tools_structure = {
        "データ処理": [
            "tools/data_processing/build_developer_network.py",
            "tools/data_processing/generate_backlog.py",
            "tools/data_processing/generate_graph.py",
            "tools/data_processing/generate_labels.py",
            "tools/data_processing/generate_profiles.py",
            "tools/data_processing/get_github_data.py",
        ],
        "分析": [
            "tools/analysis/analyze_weights.py",
            "tools/analysis/create_expert_trajectories_bot_excluded.py",
            "tools/analysis/create_expert_trajectories.py",
        ],
    }

    print("現在の構造:")
    print("ルート直下のスクリプト:")
    for category, files in root_scripts.items():
        print(f"  {category}: {len(files)}個")
        for file in files:
            print(f"    - {file}")

    print("\nscripts/ディレクトリ:")
    for category, files in scripts_dir.items():
        print(f"  {category}: {len(files)}個")
        for file in files:
            print(f"    - {file}")

    print("\ntools/ディレクトリ:")
    for category, files in tools_structure.items():
        print(f"  {category}: {len(files)}個")
        for file in files:
            print(f"    - {file}")


def propose_new_structure():
    """新しい構造を提案"""
    print("\n🏗️ 提案する新しい構造")
    print("=" * 60)

    new_structure = {
        "training/": {
            "description": "訓練関連スクリプト",
            "subdirs": {
                "gat/": [
                    "train_gat.py",  # train_collaborative_gat.py -> リネーム
                    "train_gat_standalone.py",  # train_gnn.py -> リネーム
                ],
                "irl/": [
                    "train_irl.py",  # そのまま
                    "train_irl_standalone.py",  # 新規作成予定
                ],
                "rl/": [
                    "train_rl.py",  # train_oss.py -> リネーム
                    "train_rl_standalone.py",  # 新規作成予定
                ],
            },
        },
        "pipelines/": {
            "description": "統合パイプライン",
            "files": [
                "full_pipeline.py",  # run_full_training_from_scratch.py -> リネーム
                "gat_irl_pipeline.py",  # GAT+IRLのみ
                "complete_pipeline.py",  # 全体統合
            ],
        },
        "analysis/": {
            "description": "分析・レポート生成",
            "subdirs": {
                "reports/": [
                    "training_analysis.py",  # analyze_training_results.py
                    "irl_analysis.py",  # analyze_irl_feature_weights.py + simple_irl_analysis.py
                    "gat_analysis.py",  # analyze_gat_features.py
                    "summary_report.py",  # generate_summary_report.py
                ],
                "visualization/": [
                    "plot_results.py",  # 各種プロット機能を統合
                    "interactive_dashboard.py",  # 新規作成予定
                ],
            },
        },
        "evaluation/": {
            "description": "評価・テスト",
            "files": [
                "evaluate_models.py",  # evaluate_2022_test.py -> 汎用化
                "test_features.py",  # test_feature_dimensions.py
                "benchmark.py",  # 新規作成予定
            ],
        },
        "data_processing/": {
            "description": "データ処理（toolsから移動）",
            "files": [
                "generate_graph.py",
                "generate_profiles.py",
                "generate_backlog.py",
                "build_network.py",  # build_developer_network.py
                "extract_github_data.py",  # get_github_data.py
            ],
        },
        "utils/": {
            "description": "ユーティリティ",
            "files": [
                "project_setup.py",  # 新規：プロジェクト初期化
                "data_validation.py",  # 新規：データ検証
                "config_manager.py",  # 新規：設定管理
            ],
        },
    }

    print("新しい構造:")
    for dir_name, info in new_structure.items():
        print(f"\n📁 {dir_name}")
        print(f"   目的: {info['description']}")

        if "subdirs" in info:
            for subdir, files in info["subdirs"].items():
                print(f"   📁 {subdir}")
                for file in files:
                    print(f"      - {file}")
        elif "files" in info:
            for file in info["files"]:
                print(f"   - {file}")


def create_reorganization_plan():
    """再編成計画を作成"""
    print("\n📋 再編成実行計画")
    print("=" * 60)

    # 移動・リネーム計画
    moves = [
        # GAT関連
        ("scripts/train_collaborative_gat.py", "training/gat/train_gat.py"),
        ("scripts/train_gnn.py", "training/gat/train_gat_standalone.py"),
        ("scripts/retrain_gnn_with_recent_data.py", "training/gat/retrain_gat.py"),
        # IRL関連
        ("scripts/train_irl.py", "training/irl/train_irl.py"),
        # RL関連
        ("scripts/train_oss.py", "training/rl/train_rl.py"),
        # パイプライン
        ("run_full_training_from_scratch.py", "pipelines/full_pipeline.py"),
        ("scripts/full_training_pipeline.py", "pipelines/legacy_pipeline.py"),
        ("scripts/run_complete_pipeline.py", "pipelines/complete_pipeline.py"),
        # 分析
        ("analyze_training_results.py", "analysis/reports/training_analysis.py"),
        ("analyze_irl_feature_weights.py", "analysis/reports/irl_detailed_analysis.py"),
        ("simple_irl_analysis.py", "analysis/reports/irl_simple_analysis.py"),
        ("generate_summary_report.py", "analysis/reports/summary_report.py"),
        ("analyze_gat_features.py", "analysis/reports/gat_analysis.py"),
        ("scripts/plot_gnn_graph.py", "analysis/visualization/plot_results.py"),
        # 評価
        ("scripts/evaluate_2022_test.py", "evaluation/evaluate_models.py"),
        ("test_feature_dimensions.py", "evaluation/test_features.py"),
        # データ処理
        (
            "tools/data_processing/generate_graph.py",
            "data_processing/generate_graph.py",
        ),
        (
            "tools/data_processing/generate_profiles.py",
            "data_processing/generate_profiles.py",
        ),
        (
            "tools/data_processing/generate_backlog.py",
            "data_processing/generate_backlog.py",
        ),
        (
            "tools/data_processing/build_developer_network.py",
            "data_processing/build_network.py",
        ),
        (
            "tools/data_processing/get_github_data.py",
            "data_processing/extract_github_data.py",
        ),
        (
            "tools/data_processing/generate_labels.py",
            "data_processing/generate_labels.py",
        ),
        # 分析ツール
        ("tools/analysis/analyze_weights.py", "analysis/reports/weight_analysis.py"),
        (
            "tools/analysis/create_expert_trajectories.py",
            "data_processing/create_trajectories.py",
        ),
    ]

    # 統合予定ファイル
    merges = [
        {
            "target": "analysis/reports/irl_analysis.py",
            "sources": [
                "analyze_irl_feature_weights.py",
                "simple_irl_analysis.py",
                "analyze_irl_completion.py",
                "analyze_irl_collaboration.py",
            ],
            "description": "IRL分析機能を統合",
        },
        {
            "target": "analysis/visualization/plot_results.py",
            "sources": [
                "scripts/plot_gnn_graph.py",
                # その他可視化機能を統合予定
            ],
            "description": "可視化機能を統合",
        },
    ]

    print("ファイル移動・リネーム:")
    for old_path, new_path in moves:
        print(f"  {old_path} -> {new_path}")

    print(f"\n統合予定:")
    for merge in merges:
        print(f"  {merge['target']}:")
        print(f"    目的: {merge['description']}")
        for source in merge["sources"]:
            print(f"    - {source}")


def create_unified_scripts():
    """統合された機能別スクリプトを作成"""
    print("\n🔄 統合スクリプト作成計画")
    print("=" * 60)

    unified_scripts = {
        "training/gat/train_gat.py": {
            "description": "GAT訓練（パイプライン用）",
            "features": [
                "協力ネットワーク対応",
                "複数データ形式サポート",
                "進捗表示",
                "結果保存",
            ],
        },
        "training/irl/train_irl.py": {
            "description": "IRL訓練（パイプライン用）",
            "features": ["GAT特徴量対応", "動的特徴量次元", "重み学習", "結果分析"],
        },
        "training/rl/train_rl.py": {
            "description": "RL訓練（パイプライン用）",
            "features": [
                "PPOエージェント",
                "メモリ効率化",
                "開発者数制限",
                "学習履歴保存",
            ],
        },
        "analysis/reports/irl_analysis.py": {
            "description": "IRL結果統合分析",
            "features": ["重み詳細分析", "分かりやすい解釈", "可視化", "レポート生成"],
        },
        "pipelines/full_pipeline.py": {
            "description": "完全統合パイプライン",
            "features": [
                "データ準備",
                "GAT -> IRL -> RL",
                "エラーハンドリング",
                "結果検証",
            ],
        },
    }

    for script_path, info in unified_scripts.items():
        print(f"\n📄 {script_path}")
        print(f"   目的: {info['description']}")
        print("   機能:")
        for feature in info["features"]:
            print(f"     - {feature}")


def main():
    """メイン実行"""
    print("🔍 Kazoo プロジェクト構造整理計画")
    print(f"📅 作成日時: {datetime.now()}")
    print("=" * 80)

    analyze_current_structure()
    propose_new_structure()
    create_reorganization_plan()
    create_unified_scripts()

    print("\n🎯 次のステップ:")
    print("1. 新しいディレクトリ構造を作成")
    print("2. ファイルを移動・リネーム")
    print("3. 統合スクリプトを作成")
    print("4. パスの更新")
    print("5. 動作確認")

    print("\n📋 実行確認:")
    print("この計画で進めますか？(y/n)")


if __name__ == "__main__":
    main()
