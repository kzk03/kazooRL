#!/usr/bin/env python3
"""
GAT埋め込みの直接分析
"""

from pathlib import Path

import numpy as np
import torch


def direct_embedding_analysis():
    """GAT埋め込みを直接読み込んで分析"""
    print("🔍 GAT埋め込みの直接分析")
    
    try:
        # 保存された埋め込みが存在するかチェック
        model_path = Path("data/gat_model_unified.pt")
        graph_path = Path("data/graph.pt")
        
        if not model_path.exists():
            print(f"❌ モデルファイルが見つかりません: {model_path}")
            return
            
        if not graph_path.exists():
            print(f"❌ グラフファイルが見つかりません: {graph_path}")
            return
        
        print("✅ ファイルは存在します")
        
        # グラフデータを読み込み
        graph_data = torch.load(graph_path, weights_only=False)
        print(f"グラフデータ: {type(graph_data)}")
        
        # 開発者・タスクノードの情報
        if hasattr(graph_data, 'x_dict'):
            print(f"開発者ノード特徴量: {graph_data.x_dict['dev'].shape}")
            print(f"タスクノード特徴量: {graph_data.x_dict['task'].shape}")
            
            # 初期特徴量の分析
            dev_features = graph_data.x_dict['dev'].numpy()
            task_features = graph_data.x_dict['task'].numpy()
            
            print(f"\n=== 初期特徴量の統計 ===")
            print(f"開発者特徴量（8次元）:")
            print(f"  平均: {np.mean(dev_features, axis=0)}")
            print(f"  標準偏差: {np.std(dev_features, axis=0)}")
            
            print(f"\nタスク特徴量（9次元）:")
            print(f"  平均: {np.mean(task_features, axis=0)}")
            print(f"  標準偏差: {np.std(task_features, axis=0)}")
        
        # エッジ情報
        if hasattr(graph_data, 'edge_index_dict'):
            for edge_type, edge_index in graph_data.edge_index_dict.items():
                print(f"エッジタイプ {edge_type}: {edge_index.shape[1]} 個のエッジ")
        elif ('dev', 'writes', 'task') in graph_data:
            edge_index = graph_data[('dev', 'writes', 'task')].edge_index
            print(f"dev-task エッジ: {edge_index.shape[1]} 個")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()


def explain_gat_embedding_difficulty():
    """GAT埋め込みの解釈困難性を詳しく説明"""
    print(f"\n" + "="*60)
    print("🧠 GAT埋め込みの解釈困難性について")
    print("="*60)
    
    print("\n📚 【基本的な仕組み】")
    print("1. 入力: 開発者8次元 + タスク9次元の初期特徴量")
    print("2. 処理: GATニューラルネットワーク（2層のアテンション機構）")
    print("3. 出力: 32次元の抽象的な埋め込みベクトル")
    
    print("\n🔄 【変換プロセス】")
    print("初期特徴量 → GAT層1 → アテンション → GAT層2 → アテンション → 32次元")
    print("   8/9次元    64次元      重み付き     32次元      重み付き     埋め込み")
    
    print("\n❌ 【なぜ解釈が困難か】")
    print("\n1. 非線形変換の積み重ね:")
    print("   - ReLU関数による非線形性")
    print("   - アテンション機構による動的な重み付け")
    print("   - 複数層による高次の抽象化")
    
    print("\n2. 分散表現（Distributed Representation）:")
    print("   - 1つの概念（例：Pythonスキル）が複数次元に分散")
    print("   - 1つの次元が複数概念（スキル+経験+協力度）に関与")
    print("   - 次元間の複雑な相互作用")
    
    print("\n3. 文脈依存性:")
    print("   - 周囲のノード（他の開発者・タスク）の影響を受ける")
    print("   - ネットワーク全体の構造に依存")
    print("   - 協力関係によって意味が変化")
    
    print("\n4. 学習による最適化:")
    print("   - 特定のタスク（リンク予測）に最適化されている")
    print("   - 人間の直感とは異なる特徴量表現")
    print("   - 隠れたパターンやバイアスを学習")
    
    print("\n🤔 【具体例で考える】")
    print("gat_dev_emb_24 = 0.245 という値があったとき...")
    print("❓ これは何を意味するのか？")
    print("   - Pythonスキル？ → 一部だけかも")
    print("   - リーダーシップ？ → 他の次元との組み合わせかも")
    print("   - チームワーク？ → 協力関係の影響もある")
    print("   - 経験年数？ → 非線形に変換されている")
    print("   → 単独では意味を持たない！")
    
    print("\n✅ 【現実的なアプローチ】")
    print("\n1. 統計的特徴量を活用:")
    print("   gat_similarity     → 類似度（理解しやすい）")
    print("   gat_dev_expertise  → 専門性（意味が明確）")
    print("   gat_task_popularity→ 人気度（直感的）")
    
    print("\n2. 相対的な比較:")
    print("   - 開発者Aと開発者Bの埋め込みの類似度")
    print("   - タスクXに最も適した開発者の特定")
    print("   - クラスタリングによるグループ分け")
    
    print("\n3. 全体的なパターン観察:")
    print("   - 可視化による構造の理解")
    print("   - 異常値や特異なパターンの発見")
    print("   - トレンドや変化の追跡")
    
    print("\n💡 【結論】")
    print("GAT埋め込みの個別次元を直接解釈するのは:")
    print("❌ 非現実的 - ブラックボックスの性質上不可能")
    print("✅ 代替案 - 統計特徴量 + 相対比較 + パターン観察")
    
    print(f"\n" + "="*60)


def demonstrate_embedding_usage():
    """埋め込みの実用的な使用例"""
    print(f"\n📋 実用的なGAT埋め込みの使用例")
    print("-" * 40)
    
    print("\n🎯 【推薦システム】")
    print("# 類似度計算による開発者推薦")
    print("def find_similar_developers(target_dev_embedding, all_dev_embeddings):")
    print("    similarities = cosine_similarity(target_dev_embedding, all_dev_embeddings)")
    print("    return top_k_similar_developers")
    print("→ 個別次元の意味は不要、全体の類似パターンを活用")
    
    print("\n🔍 【クラスタリング】")
    print("# 開発者のグループ分け")
    print("def cluster_developers(dev_embeddings):")
    print("    clusters = KMeans(n_clusters=5).fit(dev_embeddings)")
    print("    return developer_groups")
    print("→ 埋め込み空間での自然な分類")
    
    print("\n📊 【異常検知】")
    print("# 異常な開発者・タスクペアの検出")
    print("def detect_anomalies(dev_emb, task_emb):")
    print("    distance = euclidean_distance(dev_emb, task_emb)")
    print("    return is_anomaly(distance)")
    print("→ 通常とは異なるパターンの発見")
    
    print("\n🎨 【可視化】")
    print("# 2次元での関係性可視化")
    print("def visualize_relationships(embeddings):")
    print("    reduced = PCA(n_components=2).fit_transform(embeddings)")
    print("    plot_scatter(reduced)")
    print("→ 高次元空間の構造を低次元で観察")


def main():
    """メイン実行"""
    direct_embedding_analysis()
    explain_gat_embedding_difficulty()
    demonstrate_embedding_usage()


if __name__ == "__main__":
    main()
