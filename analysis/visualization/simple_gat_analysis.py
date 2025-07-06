#!/usr/bin/env python3
"""
GAT埋め込みベクトルの簡易解釈可能性分析
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

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


def analyze_embedding_statistics(embeddings, name="GAT埋め込み"):
    """埋め込みベクトルの統計分析"""
    print(f"\n=== {name}の統計分析 ===")
    
    # 基本統計
    print(f"形状: {embeddings.shape}")
    print(f"平均値の範囲: [{np.mean(embeddings, axis=0).min():.3f}, {np.mean(embeddings, axis=0).max():.3f}]")
    print(f"標準偏差の範囲: [{np.std(embeddings, axis=0).min():.3f}, {np.std(embeddings, axis=0).max():.3f}]")
    print(f"全体の最小値: {np.min(embeddings):.3f}")
    print(f"全体の最大値: {np.max(embeddings):.3f}")
    
    # 各次元の活用度
    dim_stds = np.std(embeddings, axis=0)
    active_dims = np.sum(dim_stds > 0.01)  # 標準偏差が0.01以上の次元
    print(f"活用されている次元数: {active_dims}/32")
    
    # 最も重要な次元
    top_5_dims = np.argsort(dim_stds)[-5:][::-1]
    print("最も変動の大きい次元（上位5つ）:")
    for i, dim in enumerate(top_5_dims):
        print(f"  {i+1}. gat_dev_emb_{dim}: std={dim_stds[dim]:.3f}")
    
    # 最も変動の小さい次元
    bottom_5_dims = np.argsort(dim_stds)[:5]
    print("最も変動の小さい次元（下位5つ）:")
    for i, dim in enumerate(bottom_5_dims):
        print(f"  {i+1}. gat_dev_emb_{dim}: std={dim_stds[dim]:.3f}")


def simple_correlation_analysis(embeddings):
    """簡易相関分析"""
    print(f"\n=== 次元間相関分析 ===")
    
    # 相関行列計算
    corr_matrix = np.corrcoef(embeddings.T)
    
    # 高相関ペア検出
    high_corr_pairs = []
    for i in range(corr_matrix.shape[0]):
        for j in range(i+1, corr_matrix.shape[1]):
            if abs(corr_matrix[i, j]) > 0.7:  # 高い相関
                high_corr_pairs.append((i, j, corr_matrix[i, j]))
    
    print(f"高相関ペア数 (|r| > 0.7): {len(high_corr_pairs)}")
    
    if high_corr_pairs:
        print("高相関ペア（上位10個）:")
        sorted_pairs = sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)
        for i, (dim1, dim2, corr) in enumerate(sorted_pairs[:10]):
            print(f"  {i+1}. gat_dev_emb_{dim1} - gat_dev_emb_{dim2}: r={corr:.3f}")
    else:
        print("高相関ペアは見つかりませんでした")
    
    # 平均的な相関の強さ
    upper_triangle = np.triu(corr_matrix, k=1)
    mean_corr = np.mean(np.abs(upper_triangle[upper_triangle != 0]))
    print(f"次元間の平均絶対相関: {mean_corr:.3f}")


def create_simple_visualization(dev_embeddings, task_embeddings, output_dir="outputs"):
    """簡易的な可視化"""
    print(f"\n=== 簡易可視化の生成 ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 次元別分散プロット
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 3, 1)
    dev_stds = np.std(dev_embeddings, axis=0)
    plt.bar(range(32), dev_stds)
    plt.title('開発者埋め込み：次元別標準偏差')
    plt.xlabel('次元')
    plt.ylabel('標準偏差')
    plt.xticks(range(0, 32, 4))
    
    plt.subplot(2, 3, 2)
    task_stds = np.std(task_embeddings, axis=0)
    plt.bar(range(32), task_stds)
    plt.title('タスク埋め込み：次元別標準偏差')
    plt.xlabel('次元')
    plt.ylabel('標準偏差')
    plt.xticks(range(0, 32, 4))
    
    # 2. 平均値プロット
    plt.subplot(2, 3, 3)
    dev_means = np.mean(dev_embeddings, axis=0)
    plt.plot(range(32), dev_means, 'o-', label='開発者', alpha=0.7)
    task_means = np.mean(task_embeddings, axis=0)
    plt.plot(range(32), task_means, 's-', label='タスク', alpha=0.7)
    plt.title('次元別平均値')
    plt.xlabel('次元')
    plt.ylabel('平均値')
    plt.legend()
    plt.xticks(range(0, 32, 4))
    
    # 3. 分布のヒストグラム（代表的な次元）
    max_var_dim = np.argmax(dev_stds)
    plt.subplot(2, 3, 4)
    plt.hist(dev_embeddings[:, max_var_dim], bins=30, alpha=0.7, label='開発者')
    plt.hist(task_embeddings[:, max_var_dim], bins=30, alpha=0.7, label='タスク')
    plt.title(f'次元{max_var_dim}の分布（最大分散）')
    plt.xlabel('値')
    plt.ylabel('頻度')
    plt.legend()
    
    # 4. 散布図（2つの重要な次元）
    sorted_dims = np.argsort(dev_stds)[-2:]
    plt.subplot(2, 3, 5)
    plt.scatter(dev_embeddings[:, sorted_dims[0]], dev_embeddings[:, sorted_dims[1]], 
               alpha=0.6, label='開発者', s=20)
    plt.scatter(task_embeddings[:, sorted_dims[0]], task_embeddings[:, sorted_dims[1]], 
               alpha=0.6, label='タスク', s=20)
    plt.xlabel(f'次元{sorted_dims[0]}')
    plt.ylabel(f'次元{sorted_dims[1]}')
    plt.title('2つの重要次元の散布図')
    plt.legend()
    
    # 5. 累積分散寄与率（簡易版）
    plt.subplot(2, 3, 6)
    dev_variances = np.var(dev_embeddings, axis=0)
    cumsum_var = np.cumsum(np.sort(dev_variances)[::-1])
    cumsum_var_ratio = cumsum_var / cumsum_var[-1]
    plt.plot(range(1, 33), cumsum_var_ratio, 'o-')
    plt.title('累積分散寄与率')
    plt.xlabel('次元数')
    plt.ylabel('累積寄与率')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gat_embedding_simple_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 簡易可視化を保存: {output_dir}/gat_embedding_simple_analysis.png")


def interpretability_summary():
    """解釈可能性についての総合的なまとめ"""
    print(f"\n" + "="*60)
    print("🤔 GAT埋め込みベクトルの解釈可能性について")
    print("="*60)
    
    print("\n❌ 【困難な理由】")
    print("1. ブラックボックス性質:")
    print("   - 各次元が何を表すかは明確でない")
    print("   - ニューラルネットワークの非線形変換の結果")
    print("   - 人間が理解しやすい概念との直接対応がない")
    
    print("\n2. 分散表現:")
    print("   - 1つの概念が複数次元に分散")
    print("   - 1つの次元が複数概念に関与")
    print("   - 次元間の複雑な相互作用")
    
    print("\n3. 学習データ依存:")
    print("   - 特定のデータセットで学習した結果")
    print("   - 隠れたバイアスやパターンを含む可能性")
    
    print("\n✅ 【活用可能なアプローチ】")
    print("1. 統計的特徴量の利用:")
    print("   - gat_similarity (類似度)")
    print("   - gat_dev_expertise (専門性)")
    print("   - gat_task_popularity (人気度)")
    
    print("\n2. 相対的比較:")
    print("   - 開発者間の類似度計算")
    print("   - クラスタリングによるグループ分け")
    print("   - 近傍探索による推薦")
    
    print("\n3. 可視化による洞察:")
    print("   - 2次元/3次元での投影")
    print("   - クラスター構造の観察")
    print("   - 異常値の検出")
    
    print("\n4. アブレーション研究:")
    print("   - 特定次元を除いた実験")
    print("   - 重要度の推定")
    print("   - 影響度の測定")
    
    print("\n💡 【推奨事項】")
    print("- 個別次元の解釈は諦める")
    print("- 全体的なパターンや関係性に注目")
    print("- 統計的特徴量と組み合わせて活用")
    print("- ドメイン知識と組み合わせた分析")


def main():
    """メイン実行関数"""
    print("🔍 GAT埋め込みベクトルの解釈可能性分析（簡易版）")
    
    # データ読み込み
    dev_embeddings, task_embeddings, extractor = load_gat_embeddings()
    
    if dev_embeddings is None:
        print("❌ データの読み込みに失敗しました")
        return
    
    # 統計分析
    analyze_embedding_statistics(dev_embeddings, "開発者埋め込み")
    analyze_embedding_statistics(task_embeddings, "タスク埋め込み")
    
    # 相関分析
    simple_correlation_analysis(dev_embeddings)
    
    # 簡易可視化
    create_simple_visualization(dev_embeddings, task_embeddings)
    
    # 解釈可能性まとめ
    interpretability_summary()
    
    print("\n✅ 分析完了！")


if __name__ == "__main__":
    main()
