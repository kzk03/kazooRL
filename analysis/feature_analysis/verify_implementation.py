#!/usr/bin/env python3
"""
特徴量分析基盤の実装確認
======================

実装したクラスが正しく動作するかを確認します。
"""

import sys
from pathlib import Path

import numpy as np

# 現在のディレクトリを追加
sys.path.append(str(Path(__file__).parent))


def main():
    """メイン関数"""
    print("🚀 特徴量分析基盤実装確認開始")
    print("=" * 60)

    # テストデータ生成
    n_samples, n_features = 100, 10
    feature_data = np.random.randn(n_samples, n_features)
    feature_names = [
        (
            f"task_feature_{i}"
            if i < 3
            else f"dev_feature_{i}" if i < 6 else f"match_feature_{i}"
        )
        for i in range(n_features)
    ]
    weights = np.random.randn(n_features) * 0.5

    print(f"✅ テストデータ生成完了: {feature_data.shape}")

    # 重みファイル保存
    weights_path = "test_weights.npy"
    np.save(weights_path, weights)

    try:
        # 1. FeatureImportanceAnalyzer のテスト
        print("\n📊 FeatureImportanceAnalyzer テスト")
        print("-" * 40)

        from feature_importance_analyzer import FeatureImportanceAnalyzer

        importance_analyzer = FeatureImportanceAnalyzer(weights_path, feature_names)
        print("✅ 初期化成功")

        importance_results = importance_analyzer.analyze_importance()
        print(f"✅ 重要度分析完了")
        print(
            f"   - 重要度ランキング: {len(importance_results['importance_ranking'])}項目"
        )
        print(
            f"   - カテゴリ別分析: {len(importance_results['category_importance'])}カテゴリ"
        )

        # TOP3を表示
        print("   重要度TOP3:")
        for i, (name, weight, importance) in enumerate(
            importance_results["importance_ranking"][:3], 1
        ):
            print(
                f"     {i}. {name:20s} | 重み:{weight:6.3f} | 重要度:{importance:6.3f}"
            )

        # 2. FeatureCorrelationAnalyzer のテスト
        print("\n🔗 FeatureCorrelationAnalyzer テスト")
        print("-" * 40)

        from feature_correlation_analyzer import FeatureCorrelationAnalyzer

        correlation_analyzer = FeatureCorrelationAnalyzer(feature_data, feature_names)
        print("✅ 初期化成功")

        correlation_results = correlation_analyzer.analyze_correlations(
            correlation_threshold=0.3
        )
        print(f"✅ 相関分析完了")
        print(f"   - 相関行列: {correlation_results['correlation_matrix'].shape}")
        print(
            f"   - 高相関ペア: {len(correlation_results['high_correlation_pairs'])}ペア"
        )
        print(
            f"   - 冗長特徴量候補: {len(correlation_results['redundant_features'])}個"
        )

        # 3. FeatureDistributionAnalyzer のテスト
        print("\n📈 FeatureDistributionAnalyzer テスト")
        print("-" * 40)

        from feature_distribution_analyzer import FeatureDistributionAnalyzer

        distribution_analyzer = FeatureDistributionAnalyzer(feature_data, feature_names)
        print("✅ 初期化成功")

        distribution_results = distribution_analyzer.analyze_distributions()
        print(f"✅ 分布分析完了")
        print(f"   - 分布統計: {len(distribution_results['distribution_stats'])}特徴量")
        print(f"   - 正規性検定: {len(distribution_results['normality_tests'])}特徴量")
        print(
            f"   - 外れ値検出: {len(distribution_results['outlier_detection'])}特徴量"
        )

        # スケール不均衡情報
        scale_info = distribution_results["scale_imbalance"]
        print(f"   - 標準偏差比（最大/最小）: {scale_info['std_ratio_max_min']:.2f}")

        print("\n🎉 全ての実装確認完了！")
        print("=" * 60)
        print("✅ FeatureImportanceAnalyzer: 動作確認済み")
        print("✅ FeatureCorrelationAnalyzer: 動作確認済み")
        print("✅ FeatureDistributionAnalyzer: 動作確認済み")
        print("\n📝 要件1.1, 1.2, 1.3の実装が完了しました。")

        return True

    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # クリーンアップ
        Path(weights_path).unlink(missing_ok=True)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
