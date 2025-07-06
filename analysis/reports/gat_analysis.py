#!/usr/bin/env python3
"""
GAT特徴量の分析 - 協力関係の埋め込み表現を解析
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append("src")


def analyze_gat_embeddings():
    """GAT埋め込み表現の分析"""
    print("🧠 GAT特徴量分析")
    print("=" * 60)

    # GATモデルの読み込み
    gat_model_path = Path("data/gnn_model_collaborative.pt")
    if not gat_model_path.exists():
        print(f"❌ GATモデルファイルが見つかりません: {gat_model_path}")
        return None

    try:
        # weights_only=Falseを指定してtorch_geometricのクラスも読み込み可能にする
        gat_model = torch.load(gat_model_path, map_location="cpu", weights_only=False)
        print(f"✅ GATモデル読み込み成功")

        if isinstance(gat_model, dict):
            print(f"📊 モデル状態辞書のキー数: {len(gat_model)}")

            # パラメータの詳細分析
            layer_info = {}
            for name, param in gat_model.items():
                if isinstance(param, torch.Tensor):
                    layer_type = name.split(".")[0] if "." in name else name
                    if layer_type not in layer_info:
                        layer_info[layer_type] = []
                    layer_info[layer_type].append((name, param.shape, param.numel()))

            print(f"\n📋 レイヤー別パラメータ情報:")
            total_params = 0
            for layer_type, params in layer_info.items():
                layer_params = sum([p[2] for p in params])
                total_params += layer_params
                print(f"  {layer_type}: {layer_params:,} parameters")
                for name, shape, count in params:
                    print(f"    - {name}: {shape}")

            print(f"\n🔢 総パラメータ数: {total_params:,}")

            # 特定のレイヤーの重み分析
            analyze_attention_weights(gat_model)

        return gat_model

    except Exception as e:
        print(f"❌ GATモデル読み込みエラー: {e}")
        return None


def analyze_attention_weights(model_dict):
    """アテンション重みの分析"""
    print(f"\n🎯 アテンション機構分析:")

    attention_layers = {}
    linear_layers = {}

    for name, param in model_dict.items():
        if "att" in name.lower() or "attention" in name.lower():
            attention_layers[name] = param
        elif "linear" in name.lower() or "lin" in name.lower():
            linear_layers[name] = param

    print(f"  - アテンション関連レイヤー数: {len(attention_layers)}")
    print(f"  - 線形変換レイヤー数: {len(linear_layers)}")

    # アテンション重みの統計
    for name, weights in attention_layers.items():
        if isinstance(weights, torch.Tensor):
            w_np = weights.detach().numpy()
            print(f"\n📊 {name}:")
            print(f"    形状: {weights.shape}")
            print(f"    平均: {w_np.mean():.6f}")
            print(f"    標準偏差: {w_np.std():.6f}")
            print(f"    最小値: {w_np.min():.6f}")
            print(f"    最大値: {w_np.max():.6f}")

    # 線形変換重みの分析
    for name, weights in linear_layers.items():
        if isinstance(weights, torch.Tensor) and len(weights.shape) == 2:
            w_np = weights.detach().numpy()
            print(f"\n📊 {name}:")
            print(f"    形状: {weights.shape}")
            print(f"    入力次元: {weights.shape[1]}")
            print(f"    出力次元: {weights.shape[0]}")
            print(f"    重み平均: {w_np.mean():.6f}")
            print(f"    重み標準偏差: {w_np.std():.6f}")


def analyze_graph_structure():
    """グラフ構造の分析"""
    print(f"\n🕸️ グラフ構造分析:")
    print("=" * 50)

    # 協力グラフの読み込み
    graph_path = Path("data/graph_collaborative.pt")
    if not graph_path.exists():
        print(f"❌ 協力グラフファイルが見つかりません: {graph_path}")
        return None

    try:
        graph_data = torch.load(graph_path, map_location="cpu", weights_only=False)
        print(f"✅ 協力グラフ読み込み成功")
        print(f"📊 グラフタイプ: {type(graph_data)}")

        if hasattr(graph_data, "x"):
            print(f"  - ノード特徴量: {graph_data.x.shape}")
            print(f"  - ノード数: {graph_data.x.shape[0]}")
            print(f"  - 各ノードの特徴量次元: {graph_data.x.shape[1]}")

            # ノード特徴量の統計
            node_features = graph_data.x.detach().numpy()
            print(f"  - ノード特徴量統計:")
            print(f"    平均: {node_features.mean():.6f}")
            print(f"    標準偏差: {node_features.std():.6f}")
            print(f"    最小値: {node_features.min():.6f}")
            print(f"    最大値: {node_features.max():.6f}")

        if hasattr(graph_data, "edge_index"):
            edge_index = graph_data.edge_index
            print(f"  - エッジインデックス: {edge_index.shape}")
            print(f"  - エッジ数: {edge_index.shape[1]}")

            # エッジの分析
            unique_nodes = torch.unique(edge_index).numpy()
            print(f"  - グラフに含まれるノード数: {len(unique_nodes)}")

            # 次数分析
            from collections import Counter

            source_nodes = edge_index[0].numpy()
            target_nodes = edge_index[1].numpy()

            source_counts = Counter(source_nodes)
            target_counts = Counter(target_nodes)

            out_degrees = list(source_counts.values())
            in_degrees = list(target_counts.values())

            print(f"  - 出次数統計:")
            print(f"    平均: {np.mean(out_degrees):.2f}")
            print(f"    最大: {max(out_degrees)}")
            print(f"    最小: {min(out_degrees)}")

            print(f"  - 入次数統計:")
            print(f"    平均: {np.mean(in_degrees):.2f}")
            print(f"    最大: {max(in_degrees)}")
            print(f"    最小: {min(in_degrees)}")

        if hasattr(graph_data, "edge_attr"):
            print(f"  - エッジ属性: {graph_data.edge_attr.shape}")
            edge_attr = graph_data.edge_attr.detach().numpy()
            print(f"  - エッジ属性統計:")
            print(f"    平均: {edge_attr.mean():.6f}")
            print(f"    標準偏差: {edge_attr.std():.6f}")

        return graph_data

    except Exception as e:
        print(f"❌ 協力グラフ読み込みエラー: {e}")
        return None


def visualize_gat_analysis(gat_model, graph_data, irl_weights):
    """GAT分析の可視化"""
    print(f"\n📊 GAT分析可視化生成中...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. IRL重みのGAT部分に焦点
    if len(irl_weights) > 25:
        gat_weights = irl_weights[25:]

        ax1.bar(range(len(gat_weights)), gat_weights, alpha=0.7)
        ax1.set_xlabel("GAT Feature Index")
        ax1.set_ylabel("IRL Weight")
        ax1.set_title("IRL Weights for GAT Features")
        ax1.grid(True, alpha=0.3)

        # 重要なGAT特徴量をハイライト
        top_gat_indices = np.argsort(np.abs(gat_weights))[-5:]
        for idx in top_gat_indices:
            ax1.bar(idx, gat_weights[idx], color="red", alpha=0.8)

    # 2. ノード次数分布（グラフ構造）
    if graph_data and hasattr(graph_data, "edge_index"):
        edge_index = graph_data.edge_index.numpy()
        degrees = np.bincount(edge_index.flatten())
        degrees = degrees[degrees > 0]  # 0次数ノードを除外

        ax2.hist(degrees, bins=30, alpha=0.7, edgecolor="black")
        ax2.set_xlabel("Node Degree")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Node Degree Distribution")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

    # 3. GAT層の重み分布（最初の線形層）
    if gat_model and isinstance(gat_model, dict):
        # 最初の線形層を探す
        first_linear = None
        for name, param in gat_model.items():
            if (
                "lin" in name.lower()
                and "weight" in name
                and isinstance(param, torch.Tensor)
            ):
                first_linear = param.detach().numpy()
                break

        if first_linear is not None:
            ax3.hist(first_linear.flatten(), bins=50, alpha=0.7, edgecolor="black")
            ax3.set_xlabel("Weight Value")
            ax3.set_ylabel("Frequency")
            ax3.set_title("GAT Layer Weight Distribution")
            ax3.grid(True, alpha=0.3)

    # 4. GAT特徴量の重要度（IRL重みの絶対値）
    if len(irl_weights) > 25:
        gat_importance = np.abs(irl_weights[25:])
        cumsum_gat = np.cumsum(np.sort(gat_importance)[::-1])
        cumsum_gat_norm = cumsum_gat / cumsum_gat[-1] * 100

        ax4.plot(range(1, len(gat_importance) + 1), cumsum_gat_norm, "g-", linewidth=2)
        ax4.axhline(80, color="red", linestyle="--", alpha=0.7, label="80%")
        ax4.axhline(95, color="orange", linestyle="--", alpha=0.7, label="95%")
        ax4.set_xlabel("Number of GAT Features")
        ax4.set_ylabel("Cumulative Importance (%)")
        ax4.set_title("GAT Feature Importance (Cumulative)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存
    output_path = (
        Path("outputs") / f"gat_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ GAT分析グラフ保存: {output_path}")
    plt.close()


def main():
    """メイン実行"""
    print("🔍 GAT特徴量詳細分析")
    print(f"📅 実行日時: {datetime.now()}")
    print("=" * 60)

    try:
        # GATモデル分析
        gat_model = analyze_gat_embeddings()

        # グラフ構造分析
        graph_data = analyze_graph_structure()

        # IRL重みの読み込み
        weights_path = Path("data/learned_weights_training.npy")
        irl_weights = None
        if weights_path.exists():
            irl_weights = np.load(weights_path)
            print(f"✅ IRL重み読み込み: {irl_weights.shape}")

        # 可視化
        if gat_model or graph_data or irl_weights is not None:
            visualize_gat_analysis(gat_model, graph_data, irl_weights)

        print(f"\n🎉 GAT分析完了!")

    except Exception as e:
        print(f"❌ 分析エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
