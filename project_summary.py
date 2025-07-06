#!/usr/bin/env python3
"""
最終プロジェクト構造サマリーとテスト
"""

import subprocess
from pathlib import Path


def show_project_summary():
    """プロジェクト構造のサマリーを表示"""
    print("🎉 Kazoo プロジェクト最終構造")
    print("=" * 60)
    
    structure = {
        "training/": {
            "description": "学習・訓練スクリプト",
            "subdirs": {
                "gat/": "Graph Attention Network関連 (6スクリプト)",
                "irl/": "Inverse Reinforcement Learning関連 (1スクリプト)",
                "rl/": "Reinforcement Learning関連 (2スクリプト)"
            }
        },
        "analysis/": {
            "description": "分析・レポート生成",
            "subdirs": {
                "reports/": "各種分析レポート (9スクリプト)",
                "visualization/": "可視化スクリプト (1スクリプト)"
            }
        },
        "pipelines/": {
            "description": "エンドツーエンドパイプライン",
            "files": [
                "unified_pipeline.py (推奨 - 単体・統合両対応)",
                "full_pipeline.py",
                "complete_pipeline.py",
                "full_training_pipeline.py"
            ]
        },
        "evaluation/": {
            "description": "モデル評価・テスト",
            "files": [
                "evaluate_models.py",
                "test_features.py",
                "evaluate_2022_test.py",
                "test_feature_dimensions.py"
            ]
        },
        "data_processing/": {
            "description": "データ前処理",
            "files": [
                "generate_graph.py",
                "generate_profiles.py", 
                "generate_labels.py",
                "generate_backlog.py",
                "build_network.py",
                "extract_github_data.py",
                "create_expert_trajectories.py"
            ]
        },
        "utils/": {
            "description": "ユーティリティ",
            "files": [
                "project_setup.py",
                "config_manager.py",
                "project_restructure_plan.py",
                "execute_restructure.py"
            ]
        },
        "src/kazoo/": {
            "description": "メインソースコード",
            "subdirs": {
                "agents/": "エージェント実装",
                "envs/": "環境定義",
                "features/": "特徴量抽出器",
                "learners/": "学習アルゴリズム",
                "GAT/": "GATモデル",
                "utils/": "共通ユーティリティ"
            }
        }
    }
    
    for folder, info in structure.items():
        print(f"\n📁 {folder}")
        print(f"   {info['description']}")
        
        if 'subdirs' in info:
            for subdir, desc in info['subdirs'].items():
                print(f"   ├── {subdir}: {desc}")
        
        if 'files' in info:
            for file_info in info['files']:
                print(f"   ├── {file_info}")

def show_usage_examples():
    """使用例を表示"""
    print("\n\n🚀 使用例")
    print("=" * 60)
    
    examples = [
        ("完全パイプライン実行", "python pipelines/unified_pipeline.py"),
        ("GAT単体実行", "python pipelines/unified_pipeline.py gat"),
        ("IRL単体実行", "python pipelines/unified_pipeline.py irl"),
        ("RL単体実行", "python pipelines/unified_pipeline.py rl"),
        ("総合分析レポート", "python analysis/reports/summary_report.py"),
        ("IRL詳細分析", "python analysis/reports/irl_analysis.py"),
        ("GAT特徴量分析", "python analysis/reports/gat_analysis.py"),
        ("協力関係分析", "python analysis/reports/collaboration_analysis.py"),
        ("モデル評価", "python evaluation/evaluate_models.py"),
        ("データ前処理", "python data_processing/generate_graph.py"),
    ]
    
    for desc, cmd in examples:
        print(f"• {desc}:")
        print(f"  {cmd}")
        print()

def count_files():
    """ファイル数をカウント"""
    print("\n📊 ファイル統計")
    print("=" * 60)
    
    root = Path("/Users/kazuki-h/rl/kazoo")
    
    folders = [
        "training",
        "analysis", 
        "pipelines",
        "evaluation",
        "data_processing",
        "utils",
        "src/kazoo"
    ]
    
    total_py_files = 0
    
    for folder in folders:
        folder_path = root / folder
        if folder_path.exists():
            py_files = list(folder_path.rglob("*.py"))
            py_count = len([f for f in py_files if "__pycache__" not in str(f)])
            total_py_files += py_count
            print(f"{folder}: {py_count} Python files")
    
    print(f"\n合計: {total_py_files} Python files")

def test_imports():
    """主要モジュールのインポートテスト"""
    print("\n🧪 基本動作テスト")
    print("=" * 60)
    
    try:
        # 基本的なインポートテスト
        import sys
        sys.path.append('src')
        
        print("✓ sys, pathlib")
        
        import numpy as np
        print("✓ numpy")
        
        import torch
        print("✓ torch")
        
        import yaml
        print("✓ yaml")
        
        import json
        print("✓ json")
        
        print("\n✅ 基本依存関係は正常です")
        
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")

def main():
    show_project_summary()
    show_usage_examples() 
    count_files()
    test_imports()
    
    print("\n" + "=" * 60)
    print("🎯 推奨開始コマンド:")
    print("python pipelines/unified_pipeline.py")
    print("\n📚 詳細ドキュメント:")
    print("README_FINAL.md")
    print("=" * 60)

if __name__ == "__main__":
    main()
