#!/usr/bin/env python3
"""
特徴量次元のテストスクリプト
"""

import sys
from pathlib import Path

from omegaconf import OmegaConf

# パスを追加
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root / "src"))

def test_feature_dimensions():
    """特徴量次元をテスト"""
    print("🔍 特徴量次元をテスト中...")
    
    try:
        from kazoo.features.feature_extractor import FeatureExtractor

        # 設定を読み込み
        cfg = OmegaConf.load("configs/base_training.yaml")
        
        # FeatureExtractorを初期化
        feature_extractor = FeatureExtractor(cfg)
        
        # 特徴量名を取得
        feature_names = feature_extractor.feature_names
        print(f"✅ FeatureExtractor初期化完了")
        print(f"   - 特徴量数: {len(feature_names)}")
        
        # GAT特徴量が含まれているかチェック
        gat_features = [name for name in feature_names if 'gat' in name]
        print(f"   - GAT特徴量数: {len(gat_features)}")
        
        if len(feature_names) > 30:
            print(f"   - 基本特徴量: {feature_names[:25]}")
            print(f"   - GAT特徴量（最初の10個）: {gat_features[:10]}")
            print(f"   - GAT特徴量（最後の5個）: {gat_features[-5:]}")
        else:
            print(f"   - 全特徴量: {feature_names}")
            
    except Exception as e:
        print(f"❌ FeatureExtractorのテスト中にエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_feature_dimensions()
