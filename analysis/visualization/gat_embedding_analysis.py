#!/usr/bin/env python3
"""
GAT埋め込みベクトルの解釈可能性分析スクリプト
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# プロジェクトルートを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from kazoo.features.gnn_feature_extractor import GNNFeatureExtractor


def load_gat_embeddings():
    """GAT埋め込みベクトルを読み込み"""
    try:
        # 設定を簡易的に作成
        class SimpleConfig:
            def __init__(self):
                self.irl = SimpleConfig()
                self.irl.use_gat = True
                self.irl.gat_graph_path = "data/graph.pt"
                self.irl.gat_model_path = "data/gat_model_unified.pt"

        cfg = SimpleConfig()

        # GAT特徴量抽出器を初期化
        extractor = GNNFeatureExtractor(cfg)

        if not extractor.model or not extractor.embeddings:
            print("❌ GAT特徴量抽出器の初期化に失敗")
            return None, None, None

        dev_embeddings = extractor.embeddings["dev"].detach().cpu().numpy()
        task_embeddings = extractor.embeddings["task"].detach().cpu().numpy()

        return dev_embeddings, task_embeddings, extractor

    except Exception as e:
        print(f"❌ エラー: {e}")
        return None, None, None


def analyze_embedding_dimensions(embeddings, name="GAT埋め込み"):
    """埋め込みベクトルの次元分析"""
    print(f"\n=== {name}の次元分析 ===")

    # 基本統計
    print(f"形状: {embeddings.shape}")
    print(f"平均値: {np.mean(embeddings, axis=0)[:5]}... (最初の5次元)")
    print(f"標準偏差: {np.std(embeddings, axis=0)[:5]}... (最初の5次元)")
    print(f"最小値: {np.min(embeddings)}")
    print(f"最大値: {np.max(embeddings)}")

    # 次元間の相関分析
    corr_matrix = np.corrcoef(embeddings.T)
    high_corr_pairs = []

    for i in range(corr_matrix.shape[0]):
        for j in range(i + 1, corr_matrix.shape[1]):
            if abs(corr_matrix[i, j]) > 0.7:  # 高い相関
                high_corr_pairs.append((i, j, corr_matrix[i, j]))

    print(f"高相関ペア数 (|r| > 0.7): {len(high_corr_pairs)}")
    if high_corr_pairs:
        print("高相関ペア（上位5個）:")
        for i, j, corr in sorted(
            high_corr_pairs, key=lambda x: abs(x[2]), reverse=True
        )[:5]:
            print(f"  次元{i} - 次元{j}: r={corr:.3f}")


def visualize_embeddings_2d(dev_embeddings, task_embeddings, output_dir="outputs"):
    """埋め込みベクトルの2次元可視化"""
    print("\n=== 2次元可視化の生成 ===")

    os.makedirs(output_dir, exist_ok=True)

    # PCA
    print("PCA実行中...")
    pca = PCA(n_components=2)
    all_embeddings = np.vstack([dev_embeddings, task_embeddings])
    pca_result = pca.fit_transform(all_embeddings)

    dev_pca = pca_result[: len(dev_embeddings)]
    task_pca = pca_result[len(dev_embeddings) :]

    plt.figure(figsize=(12, 5))

    # PCAプロット
    plt.subplot(1, 2, 1)
    plt.scatter(dev_pca[:, 0], dev_pca[:, 1], alpha=0.6, label="開発者", s=20)
    plt.scatter(task_pca[:, 0], task_pca[:, 1], alpha=0.6, label="タスク", s=20)
    plt.xlabel(f"PC1 (分散比: {pca.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"PC2 (分散比: {pca.explained_variance_ratio_[1]:.2f})")
    plt.title("GAT埋め込み - PCA")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # t-SNE（サンプル数を制限）
    print("t-SNE実行中...")
    sample_size = min(500, len(all_embeddings))  # 計算時間を短縮
    sample_indices = np.random.choice(len(all_embeddings), sample_size, replace=False)
    sample_embeddings = all_embeddings[sample_indices]

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, sample_size // 4))
    tsne_result = tsne.fit_transform(sample_embeddings)

    dev_indices = sample_indices[sample_indices < len(dev_embeddings)]
    task_indices = sample_indices[sample_indices >= len(dev_embeddings)] - len(
        dev_embeddings
    )

    plt.subplot(1, 2, 2)
    if len(dev_indices) > 0:
        dev_tsne_indices = np.where(sample_indices < len(dev_embeddings))[0]
        plt.scatter(
            tsne_result[dev_tsne_indices, 0],
            tsne_result[dev_tsne_indices, 1],
            alpha=0.6,
            label="開発者",
            s=20,
        )

    if len(task_indices) > 0:
        task_tsne_indices = np.where(sample_indices >= len(dev_embeddings))[0]
        plt.scatter(
            tsne_result[task_tsne_indices, 0],
            tsne_result[task_tsne_indices, 1],
            alpha=0.6,
            label="タスク",
            s=20,
        )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("GAT埋め込み - t-SNE")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/gat_embeddings_2d.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ 2次元可視化を保存: {output_dir}/gat_embeddings_2d.png")

    return pca


def analyze_dimension_importance(pca, embeddings, top_n=10):
    """PCA成分による次元重要度分析"""
    print(f"\n=== 次元重要度分析（上位{top_n}次元） ===")

    # 第1主成分への貢献度
    pc1_importance = np.abs(pca.components_[0])
    top_dims_pc1 = np.argsort(pc1_importance)[-top_n:][::-1]

    print("第1主成分への貢献度が高い次元:")
    for i, dim in enumerate(top_dims_pc1):
        print(f"  {i+1}. gat_dev_emb_{dim}: {pc1_importance[dim]:.3f}")

    # 第2主成分への貢献度
    pc2_importance = np.abs(pca.components_[1])
    top_dims_pc2 = np.argsort(pc2_importance)[-top_n:][::-1]

    print("\n第2主成分への貢献度が高い次元:")
    for i, dim in enumerate(top_dims_pc2):
        print(f"  {i+1}. gat_dev_emb_{dim}: {pc2_importance[dim]:.3f}")


def create_interpretability_report(dev_embeddings, task_embeddings):
    """解釈可能性レポートの生成"""
    print("\n=== 解釈可能性レポート ===")

    # 統計的要約
    print("📊 統計的要約:")
    print(
        f"  - 開発者埋め込み: {dev_embeddings.shape[0]}個 × {dev_embeddings.shape[1]}次元"
    )
    print(
        f"  - タスク埋め込み: {task_embeddings.shape[0]}個 × {task_embeddings.shape[1]}次元"
    )

    # 分散の説明
    total_var_dev = np.var(dev_embeddings, axis=0).sum()
    print(f"  - 開発者埋め込みの総分散: {total_var_dev:.2f}")

    # 次元の活用度
    zero_dims_dev = np.sum(np.std(dev_embeddings, axis=0) < 0.01)
    print(f"  - ほぼ使われていない次元: {zero_dims_dev}/32")

    # 解釈の難しさ
    print("\n🤔 解釈可能性:")
    print("  - 各次元の直接的な意味: ❌ 不明（ブラックボックス）")
    print("  - 統計的特徴量: ✅ 理解可能")
    print("  - 2次元可視化: ✅ 全体的なパターンは観察可能")
    print("  - 主成分分析: ✅ 重要な次元の特定は可能")

    print("\n💡 推奨事項:")
    print("  1. 直接的な解釈ではなく、統計的特徴量を活用")
    print("  2. クラスタリングによるパターン発見")
    print("  3. 類似度計算による相対的な比較")
    print("  4. アブレーション研究による重要度分析")


def main():
    """メイン実行関数"""
    print("🔍 GAT埋め込みベクトルの解釈可能性分析")

    # データ読み込み
    dev_embeddings, task_embeddings, extractor = load_gat_embeddings()

    if dev_embeddings is None:
        print("❌ データの読み込みに失敗しました")
        return

    # 次元分析
    analyze_embedding_dimensions(dev_embeddings, "開発者埋め込み")
    analyze_embedding_dimensions(task_embeddings, "タスク埋め込み")

    # 2次元可視化
    pca = visualize_embeddings_2d(dev_embeddings, task_embeddings)

    # 重要度分析
    analyze_dimension_importance(pca, dev_embeddings)

    # 解釈可能性レポート
    create_interpretability_report(dev_embeddings, task_embeddings)

    print("\n✅ 分析完了！")


if __name__ == "__main__":
    main()
