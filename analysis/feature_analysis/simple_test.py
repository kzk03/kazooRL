#!/usr/bin/env python3
"""
特徴量分析基盤の簡単なテスト
==========================

基本的な動作確認を行います。
"""

import sys
from pathlib import Path

import numpy as np

# プロジェクトルートを追加
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))


def test_basic_functionality():
    """基本機能のテスト"""
    print("🧪 基本機能テスト開始")

    # テストデータ生成
    n_samples, n_features = 100, 10
    feature_data = np.random.randn(n_samples, n_features)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    weights = np.random.randn(n_features)

    print(f"✅ テストデータ生成完了: {feature_data.shape}")

    # 重みファイル保存
    weights_path = "test_weights.npy"
    np.save(weights_path, weights)

    try:
        # FeatureImportanceAnalyzer のテスト
        from kazoo.analysis.feature_analysis.feature_importance_analyzer import \
            FeatureImportanceAnalyzer

        analyzer = FeatureImportanceAnalyzer(weights_path, feature_names)
        print("✅ FeatureImportanceAnalyzer 初期化成功")

        # 基本的な分析実行
        results = analyzer.analyze_importance()
        print(f"✅ 重要度分析完了: {len(results['importance_ranking'])}項目")

        # FeatureCorrelationAnalyzer のテスト
        from kazoo.analysis.feature_analysis.feature_correlation_analyzer import \
            FeatureCorrelationAnalyzer

        corr_analyzer = FeatureCorrelationAnalyzer(feature_data, feature_names)
        print("✅ FeatureCorrelationAnalyzer 初期化成功")

        corr_results = corr_analyzer.analyze_correlations()
        print(f"✅ 相関分析完了: {corr_results['correlation_matrix'].shape}")

        # FeatureDistributionAnalyzer のテスト
        from kazoo.analysis.feature_analysis.feature_distribution_analyzer import \
            FeatureDistributionAnalyzer

        dist_analyzer = FeatureDistributionAnalyzer(feature_data, feature_names)
        print("✅ FeatureDistributionAnalyzer 初期化成功")

        dist_results = dist_analyzer.analyze_distributions()
        print(f"✅ 分布分析完了: {len(dist_results['distribution_stats'])}特徴量")

        print("\n🎉 全ての基本機能テスト完了！")

    finally:
        # クリーンアップ
        Path(weights_path).unlink(missing_ok=True)


if __name__ == "__main__":
    test_basic_functionality()
