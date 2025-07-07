#!/usr/bin/env python3
"""
GAT特徴量の詳細分析
「統計出してるやつ」と「32次元埋め込み」の違いを明確化

GAT特徴量は大きく2つのカテゴリに分かれる：
1. 解釈可能な統計特徴量（3-5次元）：明確な意味を持つ
2. GAT埋め込み特徴量（32次元）：学習された抽象的表現
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# プロジェクトパスを追加
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root / "src"))

from datetime import datetime
import os

def analyze_gat_features():
    """GAT特徴量の詳細分析"""
    
    print("=" * 80)
    print("GAT特徴量詳細分析：「統計出してるやつ」vs「32次元埋め込み」")
    print("=" * 80)
    
    # IRL重みファイルを読み込み
    weights_path = project_root / "data" / "learned_weights_training.npy"
    if not weights_path.exists():
        print(f"❌ 重みファイルが見つかりません: {weights_path}")
        return
    
    weights = np.load(weights_path)
    print(f"✅ IRL重み読み込み完了: {len(weights)} 次元")
    
    # GAT特徴量名を定義（実装に基づく）
    gat_statistical_features = [
        'gat_similarity',           # 開発者-タスク間の類似度
        'gat_dev_expertise',        # 開発者の専門性スコア
        'gat_task_popularity',      # タスクの人気度スコア
        'gat_collaboration_strength', # 協力ネットワーク内での重要度
        'gat_network_centrality'    # ネットワーク中心性
    ]
    
    gat_embedding_features = [f'gat_dev_emb_{i}' for i in range(32)]
    
    print(f"GAT統計特徴量数: {len(gat_statistical_features)}")
    print(f"GAT埋め込み特徴量数: {len(gat_embedding_features)}")
    
    # 基本特徴量の数を推定（実際の特徴量リストから）
    total_features = len(weights)
    num_gat_features = len(gat_statistical_features) + len(gat_embedding_features)
    num_basic_features = total_features - num_gat_features
    
    print(f"総特徴量数: {total_features}")
    print(f"基本特徴量数: {num_basic_features}")
    print(f"GAT特徴量数: {num_gat_features}")
    
    # GAT特徴量の重みを抽出
    gat_start_idx = num_basic_features
    gat_stat_weights = weights[gat_start_idx:gat_start_idx + len(gat_statistical_features)]
    gat_emb_weights = weights[gat_start_idx + len(gat_statistical_features):gat_start_idx + num_gat_features]
    
    print("\n" + "=" * 60)
    print("GAT特徴量の構成分析")
    print("=" * 60)
    
    print(f"\n📊 【統計特徴量（解釈可能）】 - 「統計出してるやつ」")
    print(f"特徴量数: {len(gat_statistical_features)}")
    print("-" * 50)
    
    feature_explanations = {
        'gat_similarity': '開発者-タスク間の類似度（コサイン類似度）',
        'gat_dev_expertise': '開発者の専門性スコア（類似タスクTop10との平均類似度）', 
        'gat_task_popularity': 'タスクの人気度スコア（類似開発者Top10との平均類似度）',
        'gat_collaboration_strength': '開発者の協力ネットワーク内での重要度',
        'gat_network_centrality': '開発者のネットワーク中心性（接続数ベース）'
    }
    
    for i, (feature_name, weight) in enumerate(zip(gat_statistical_features, gat_stat_weights), 1):
        explanation = feature_explanations.get(feature_name, '詳細な説明未定義')
        print(f"{i:2d}. {feature_name:25s} | 重み: {weight:8.4f} | {explanation}")
    
    if len(gat_stat_weights) > 0:
        print(f"\n📈 統計特徴量の重み統計:")
        print(f"   平均: {np.mean(gat_stat_weights):8.4f}")
        print(f"   標準偏差: {np.std(gat_stat_weights):8.4f}")
        print(f"   最大: {np.max(gat_stat_weights):8.4f} ({gat_statistical_features[np.argmax(gat_stat_weights)]})")
        print(f"   最小: {np.min(gat_stat_weights):8.4f} ({gat_statistical_features[np.argmin(gat_stat_weights)]})")
    
    print(f"\n🧠 【GAT埋め込み特徴量（32次元）】 - 「32次元埋め込み」")
    print(f"特徴量数: {len(gat_embedding_features)}")
    print("-" * 50)
    
    if len(gat_emb_weights) > 0:
        print(f"📈 GAT埋め込みの重み統計:")
        print(f"   平均: {np.mean(gat_emb_weights):8.4f}")
        print(f"   標準偏差: {np.std(gat_emb_weights):8.4f}")
        print(f"   最大: {np.max(gat_emb_weights):8.4f} ({gat_embedding_features[np.argmax(gat_emb_weights)]})")
        print(f"   最小: {np.min(gat_emb_weights):8.4f} ({gat_embedding_features[np.argmin(gat_emb_weights)]})")
        
        # 重要な埋め込み次元を特定
        print(f"\n� 重要度上位10次元:")
        emb_importance = [(i, name, abs(weight)) for i, (name, weight) in enumerate(zip(gat_embedding_features, gat_emb_weights))]
        emb_importance.sort(key=lambda x: x[2], reverse=True)
        
        for rank, (idx, name, abs_weight) in enumerate(emb_importance[:10], 1):
            dimension = name.split('_')[-1]
            actual_weight = gat_emb_weights[idx]
            print(f"{rank:2d}. {name:20s} | 重み: {actual_weight:8.4f} | 埋め込み次元{dimension}")
        
        # 正負の重みの分布
        positive_weights = gat_emb_weights[gat_emb_weights > 0]
        negative_weights = gat_emb_weights[gat_emb_weights < 0]
        zero_weights = gat_emb_weights[gat_emb_weights == 0]
        
        print(f"\n📊 重みの符号分布:")
        print(f"   正の重み: {len(positive_weights):2d}個 (平均: {np.mean(positive_weights) if len(positive_weights) > 0 else 0:6.4f})")
        print(f"   負の重み: {len(negative_weights):2d}個 (平均: {np.mean(negative_weights) if len(negative_weights) > 0 else 0:6.4f})")
        print(f"   ゼロ重み: {len(zero_weights):2d}個")
    
    print(f"\n" + "=" * 60)
    print("GAT特徴量の役割と意味の比較")
    print("=" * 60)
    
    print("""
📊 【統計特徴量（解釈可能）】の特徴:
  ✅ 明確な意味: 各特徴量が何を測定しているか明確
  ✅ 解釈しやすい: 人間が理解・デバッグしやすい
  ✅ ドメイン知識: GitHubの開発活動に関する明確な指標
  ✅ ルール制御: 明示的なルールや閾値を設定しやすい
  
  例: gat_similarity = 0.8 → 開発者とタスクが非常に類似
      gat_dev_expertise = 0.6 → 開発者の専門性が中程度
      
🧠 【GAT埋め込み（32次元）】の特徴:
  ✅ 高表現力: 複雑なパターンや関係性を学習可能
  ✅ 非線形特徴: 人間には理解困難な抽象的表現
  ✅ グラフ構造: 開発者-タスク関係をGraphAttentionで学習
  ✅ 適応性: データから自動的に重要な特徴を発見
  
  例: gat_dev_emb_15 = -0.234 → 具体的意味は不明だが、
                               IRLにとって重要なパターンを表現
""")

    # GAT特徴量の重要性をIRLの観点で分析
    if len(gat_stat_weights) > 0 and len(gat_emb_weights) > 0:
        stat_weights_abs = np.abs(gat_stat_weights)
        emb_weights_abs = np.abs(gat_emb_weights)
        
        avg_stat_importance = np.mean(stat_weights_abs)
        avg_emb_importance = np.mean(emb_weights_abs)
        
        print(f"\n🎯 IRLにおける重要性比較:")
        print(f"統計特徴量の平均重要度: {avg_stat_importance:.4f}")
        print(f"埋め込み特徴量の平均重要度: {avg_emb_importance:.4f}")
        
        if avg_emb_importance > avg_stat_importance:
            ratio = avg_emb_importance / avg_stat_importance
            print(f"→ GAT埋め込みが {ratio:.2f}倍 重要視されている")
            print("  IRLは抽象的な埋め込み表現により強く依存")
        else:
            ratio = avg_stat_importance / avg_emb_importance
            print(f"→ 統計特徴量が {ratio:.2f}倍 重要視されている")
            print("  IRLは解釈可能な統計指標により強く依存")
    
    print(f"\n" + "=" * 60)
    print("システム設計への示唆")
    print("=" * 60)
    
    print("""
� 【統計特徴量】の活用方針:
  • デバッグ・検証: 推薦理由の説明に使用
  • ルールベース制御: 明示的な制約条件の設定
  • 特徴量エンジニアリング: 新しい統計指標の追加
  • 可視化・分析: 開発者-タスク関係の理解

🚀 【GAT埋め込み】の活用方針:
  • 性能向上: 推薦精度の向上に重要
  • パターン発見: 複雑な関係性の自動学習
  • スケーラビリティ: 大規模データに対応
  • 継続学習: 新しいデータでの表現更新

💡 バランス戦略:
  • 解釈性が必要 → 統計特徴量を重視
  • 性能が最優先 → GAT埋め込みを重視
  • 実運用では両者のバランスが重要
""")

    # 分析結果をCSVで保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 統計特徴量のCSV
    if len(gat_stat_weights) > 0:
        stat_data = []
        for feature_name, weight in zip(gat_statistical_features, gat_stat_weights):
            stat_data.append({
                'feature_name': feature_name,
                'weight': weight,
                'abs_weight': abs(weight),
                'explanation': feature_explanations.get(feature_name, '詳細説明未定義'),
                'category': 'GAT統計特徴量'
            })
        
        stat_df = pd.DataFrame(stat_data)
        stat_csv_path = project_root / "outputs" / f"gat_statistical_features_{timestamp}.csv"
        stat_df.to_csv(stat_csv_path, index=False, encoding='utf-8')
        print(f"\n💾 統計特徴量CSV保存: {stat_csv_path}")
    
    # GAT埋め込み特徴量のCSV
    if len(gat_emb_weights) > 0:
        emb_data = []
        for i, (feature_name, weight) in enumerate(zip(gat_embedding_features, gat_emb_weights)):
            emb_data.append({
                'feature_name': feature_name,
                'weight': weight,
                'abs_weight': abs(weight),
                'dimension': feature_name.split('_')[-1],
                'rank_by_importance': 0,  # 後で設定
                'category': 'GAT埋め込み特徴量'
            })
        
        emb_df = pd.DataFrame(emb_data)
        # 重要度でランキング
        emb_df = emb_df.sort_values('abs_weight', ascending=False).reset_index(drop=True)
        emb_df['rank_by_importance'] = range(1, len(emb_df) + 1)
        
        emb_csv_path = project_root / "outputs" / f"gat_embedding_features_{timestamp}.csv"
        emb_df.to_csv(emb_csv_path, index=False, encoding='utf-8')
        print(f"💾 埋め込み特徴量CSV保存: {emb_csv_path}")
    
    # 統合サマリーのCSV
    summary_data = []
    
    if len(gat_stat_weights) > 0:
        summary_data.append({
            'feature_type': 'GAT統計特徴量',
            'count': len(gat_stat_weights),
            'avg_weight': np.mean(np.abs(gat_stat_weights)),
            'max_weight': np.max(np.abs(gat_stat_weights)),
            'min_weight': np.min(np.abs(gat_stat_weights)),
            'description': '解釈可能な統計指標（類似度、専門性、人気度など）'
        })
    
    if len(gat_emb_weights) > 0:
        summary_data.append({
            'feature_type': 'GAT埋め込み特徴量',
            'count': len(gat_emb_weights),
            'avg_weight': np.mean(np.abs(gat_emb_weights)),
            'max_weight': np.max(np.abs(gat_emb_weights)),
            'min_weight': np.min(np.abs(gat_emb_weights)),
            'description': 'GATで学習された32次元の抽象的表現'
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = project_root / "outputs" / f"gat_features_summary_{timestamp}.csv"
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
        print(f"💾 GAT特徴量サマリーCSV保存: {summary_csv_path}")
    
    print(f"\n✅ GAT特徴量詳細分析完了")
    print(f"時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    analyze_gat_features()
    print()
    print("特徴量抽出:")
    print("  - _get_full_gnn_features()で統計+埋め込みを結合")
    print("  - features.extend(dev_emb.tolist())")
    print("  - 合計: 3-5(統計) + 32(埋め込み) = 35-37次元")

def analyze_embedding_importance():
    """埋め込み次元の重要度を詳細分析"""
    
    print(f"\n\n🔍 32次元埋め込みの詳細重要度分析")
    print("=" * 60)
    
    # GAT特徴量のCSVを読み込み
    gat_df = pd.read_csv("outputs/gat_feature_analysis_20250707_105056.csv")
    
    # feature_で始まる埋め込み次元のみ抽出
    embeddings = gat_df[gat_df['feature_name'].str.startswith('feature_')]
    
    print(f"埋め込み次元数: {len(embeddings)}")
    
if __name__ == "__main__":
    analyze_gat_features()
