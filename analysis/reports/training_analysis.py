#!/usr/bin/env python3
"""
学習結果の分析スクリプト
GAT, IRL の結果を分析します
"""

import os
import sys

sys.path.append("src")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def analyze_gat_model():
    """GAT モデルの分析"""
    print("🔍 GAT モデル分析")
    print("=" * 50)

    # GAT モデルの読み込み
    gat_model_path = "data/gnn_model_collaborative.pt"
    if not os.path.exists(gat_model_path):
        print(f"❌ GAT モデルが見つかりません: {gat_model_path}")
        return

    try:
        gat_data = torch.load(gat_model_path, map_location="cpu")
        print(f"✅ GAT モデル読み込み成功")

        # モデルの構造を表示
        if "model_state_dict" in gat_data:
            state_dict = gat_data["model_state_dict"]
            print(f"📊 モデルパラメータ数: {len(state_dict)}")

            # 各レイヤーのサイズを表示
            for key, tensor in state_dict.items():
                if "weight" in key:
                    print(f"  {key}: {tensor.shape}")

        # 学習履歴があれば表示
        if "training_history" in gat_data:
            history = gat_data["training_history"]
            print(f"📈 学習履歴: {len(history)} エポック")
            if len(history) > 0:
                print(f"  最初のロス: {history[0]:.4f}")
                print(f"  最後のロス: {history[-1]:.4f}")
                print(f"  ロス改善: {history[0] - history[-1]:.4f}")

    except Exception as e:
        print(f"❌ GAT モデル読み込みエラー: {e}")


def analyze_graph_data():
    """グラフデータの分析"""
    print("\n🔍 グラフデータ分析")
    print("=" * 50)

    # トレーニング用グラフ
    training_graph_path = "data/graph_training.pt"
    if os.path.exists(training_graph_path):
        try:
            training_graph = torch.load(
                training_graph_path, map_location="cpu", weights_only=False
            )
            print(f"✅ トレーニンググラフ読み込み成功")
            print(f"📊 トレーニンググラフ構造:")
            print(f"  開発者ノード: {training_graph['dev']['x'].shape[0]} 個")
            print(f"  タスクノード: {training_graph['task']['x'].shape[0]} 個")
            print(f"  開発者特徴量次元: {training_graph['dev']['x'].shape[1]}")
            print(f"  タスク特徴量次元: {training_graph['task']['x'].shape[1]}")

            # エッジ情報
            for edge_type, edge_data in training_graph.items():
                if "edge_index" in str(type(edge_data)) or (
                    isinstance(edge_data, dict) and "edge_index" in edge_data
                ):
                    if isinstance(edge_data, dict) and "edge_index" in edge_data:
                        edge_count = edge_data["edge_index"].shape[1]
                        print(f"  {edge_type} エッジ: {edge_count} 個")

        except Exception as e:
            print(f"❌ トレーニンググラフ読み込みエラー: {e}")

    # 協力ネットワークグラフ
    collab_graph_path = "data/graph_collaborative.pt"
    if os.path.exists(collab_graph_path):
        try:
            collab_graph = torch.load(
                collab_graph_path, map_location="cpu", weights_only=False
            )
            print(f"\n✅ 協力ネットワークグラフ読み込み成功")
            print(f"📊 協力ネットワークグラフ構造:")
            print(f"  開発者ノード: {collab_graph['dev']['x'].shape[0]} 個")
            print(f"  タスクノード: {collab_graph['task']['x'].shape[0]} 個")
            print(f"  開発者特徴量次元: {collab_graph['dev']['x'].shape[1]}")
            print(f"  タスク特徴量次元: {collab_graph['task']['x'].shape[1]}")

        except Exception as e:
            print(f"❌ 協力ネットワークグラフ読み込みエラー: {e}")


def analyze_irl_weights():
    """IRL で学習された重みの分析"""
    print("\n🔍 IRL 学習重み分析")
    print("=" * 50)

    weights_path = "data/learned_weights_training.npy"
    if not os.path.exists(weights_path):
        print(f"❌ IRL 重みが見つかりません: {weights_path}")
        return

    try:
        weights = np.load(weights_path)
        print(f"✅ IRL 重み読み込み成功")
        print(f"📊 重みの形状: {weights.shape}")
        print(f"📊 重みの統計:")
        print(f"  平均: {weights.mean():.6f}")
        print(f"  標準偏差: {weights.std():.6f}")
        print(f"  最小値: {weights.min():.6f}")
        print(f"  最大値: {weights.max():.6f}")

        # 上位・下位の重要な特徴量を表示
        sorted_indices = np.argsort(np.abs(weights))[::-1]
        print(f"\n📈 重要度の高い特徴量（上位10個）:")
        for i in range(min(10, len(weights))):
            idx = sorted_indices[i]
            print(f"  特徴量 {idx}: {weights[idx]:.6f}")

        print(f"\n📉 重要度の低い特徴量（下位5個）:")
        for i in range(max(0, len(weights) - 5), len(weights)):
            idx = sorted_indices[i]
            print(f"  特徴量 {idx}: {weights[idx]:.6f}")

    except Exception as e:
        print(f"❌ IRL 重み読み込みエラー: {e}")


def check_feature_dimensions():
    """特徴量次元の確認"""
    print("\n🔍 特徴量次元確認")
    print("=" * 50)

    try:
        sys.path.append("src")
        import yaml

        with open("configs/base_training.yaml", "r") as f:
            config = yaml.safe_load(f)

        from src.kazoo.features.feature_extractor import FeatureExtractor

        extractor = FeatureExtractor(config)

        print(f"✅ FeatureExtractor 初期化成功")
        print(f"📊 総特徴量数: {extractor.feature_dim}")
        print(f"📊 基本特徴量数: {len(extractor.feature_names)}")

        if hasattr(extractor, "gnn_extractor") and extractor.gnn_extractor is not None:
            print(f"📊 GAT特徴量数: {extractor.gnn_extractor.feature_dim}")
            print(f"✅ GAT統合: 有効")
        else:
            print(f"❌ GAT統合: 無効")

    except Exception as e:
        print(f"❌ 特徴量次元確認エラー: {e}")


def main():
    """メイン実行関数"""
    print("🎯 Kazoo 学習結果分析")
    print("=" * 60)

    analyze_gat_model()
    analyze_graph_data()
    analyze_irl_weights()
    check_feature_dimensions()

    print("\n" + "=" * 60)
    print("🎉 分析完了!")


if __name__ == "__main__":
    main()
