#!/usr/bin/env python3
"""
設定ファイルを使ったGNN特徴量抽出器の初期化テスト
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# プロジェクトのルートをパスに追加
sys.path.append(str(Path(__file__).resolve().parents[1]))

from kazoo.features.gnn_feature_extractor import GNNFeatureExtractor


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """設定ファイルを使ったGNN特徴量抽出器のテスト"""
    
    print("=" * 60)
    print("🧪 設定ファイル使用 - GNN特徴量抽出器テスト")
    print("=" * 60)
    
    # 設定内容確認
    print("\n📋 設定確認:")
    print(f"  - GNN使用: {cfg.irl.get('use_gnn', False)}")
    print(f"  - オンライン学習: {cfg.irl.get('online_gnn_learning', False)}")
    print(f"  - グラフパス: {cfg.irl.get('gnn_graph_path', 'N/A')}")
    print(f"  - モデルパス: {cfg.irl.get('gnn_model_path', 'N/A')}")
    
    # ファイル存在確認
    import os
    graph_path = cfg.irl.get('gnn_graph_path', '')
    model_path = cfg.irl.get('gnn_model_path', '')
    
    print(f"\n📁 ファイル存在確認:")
    print(f"  - グラフファイル: {graph_path} -> {'✅' if os.path.exists(graph_path) else '❌'}")
    print(f"  - モデルファイル: {model_path} -> {'✅' if os.path.exists(model_path) else '❌'}")
    
    # GNNFeatureExtractor初期化
    print(f"\n🚀 GNNFeatureExtractor初期化中...")
    try:
        gnn_extractor = GNNFeatureExtractor(cfg)
        
        if gnn_extractor.model:
            print("✅ GNN特徴量抽出器が正常に初期化されました")
            print(f"  - オンライン学習: {'有効' if gnn_extractor.online_learning else '無効'}")
            print(f"  - 開発者ノード数: {len(gnn_extractor.dev_id_to_idx)}")
            print(f"  - タスクノード数: {len(gnn_extractor.task_id_to_idx)}")
            print(f"  - 更新頻度: {gnn_extractor.update_frequency}")
            print(f"  - 学習率: {gnn_extractor.learning_rate}")
            print(f"  - バッファサイズ: {gnn_extractor.max_buffer_size}")
        else:
            print("❌ GNNモデルの読み込みに失敗")
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
