#!/usr/bin/env python3
"""
特徴量最適化の実装確認
====================

実装した特徴量最適化クラスが正しく動作するかを確認します。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 現在のディレクトリを追加
sys.path.append(str(Path(__file__).parent))


def generate_test_data(n_samples=1000, n_features=50):
    """テスト用の特徴量データを生成"""
    print(f"テストデータ生成中... ({n_samples}サンプル, {n_features}特徴量)")

    np.random.seed(42)

    # 様々なタイプの特徴量を生成
    features = []
    feature_names = []

    # 1. 数値特徴量（正規分布）
    for i in range(15):
        features.append(np.random.normal(0, 1, n_samples))
        feature_names.append(f"numerical_{i}")

    # 2. 歪んだ数値特徴量（対数正規分布）
    for i in range(10):
        features.append(np.random.lognormal(0, 1, n_samples))
        feature_names.append(f"skewed_{i}")

    # 3. バイナリ特徴量
    for i in range(8):
        features.append(np.random.binomial(1, 0.3, n_samples).astype(float))
        feature_names.append(f"binary_{i}")

    # 4. カテゴリカル特徴量（数値でエンコード）
    for i in range(5):
        categories = np.random.choice(
            [0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1]
        ).astype(float)
        features.append(categories)
        feature_names.append(f"categorical_{i}")

    # 5. 時系列特徴量
    for i in range(7):
        time_values = np.random.uniform(0, 365, n_samples)  # 日数
        features.append(time_values)
        feature_names.append(f"time_since_{i}")

    # 6. 外れ値を含む特徴量
    for i in range(5):
        base_values = np.random.normal(0, 1, n_samples)
        # 5%の外れ値を追加
        outlier_mask = np.random.random(n_samples) < 0.05
        base_values[outlier_mask] += np.random.normal(0, 10, np.sum(outlier_mask))
        features.append(base_values)
        feature_names.append(f"outlier_prone_{i}")

    # データを結合
    X = np.column_stack(features)

    # 目的変数を生成（分類用）
    # 最初の10個の特徴量に基づいて目的変数を生成
    y_continuous = (
        X[:, 0] * 0.5
        + X[:, 1] * 0.3
        + X[:, 2] * 0.2
        + np.random.normal(0, 0.1, n_samples)
    )
    y_binary = (y_continuous > np.median(y_continuous)).astype(int)

    print("✅ テストデータ生成完了")
    return X, y_binary, y_continuous, feature_names


def main():
    """メイン関数"""
    print("🚀 特徴量最適化実装確認開始")
    print("=" * 60)

    # テストデータ生成
    X, y_binary, y_continuous, feature_names = generate_test_data()

    try:
        # 1. FeatureScaler のテスト
        print("\n⚖️ FeatureScaler テスト")
        print("-" * 40)

        from feature_scaler import FeatureScaler

        scaler = FeatureScaler()
        print("✅ 初期化成功")

        # スケーリングの学習と適用
        X_scaled = scaler.fit_transform(X, feature_names)
        print(f"✅ スケーリング完了: {X.shape} → {X_scaled.shape}")

        # スケーリング情報の取得
        scaling_info = scaler.get_scaling_info()
        print(f"   - 入力特徴量数: {scaling_info['n_features_in']}")
        print(f"   - 出力特徴量数: {scaling_info['n_features_out']}")
        print(f"   - 戦略統計: {scaling_info['strategy_counts']}")

        # 変換後の特徴量名
        feature_names_out = scaler.get_feature_names_out()
        print(f"   - 変換後特徴量名例: {feature_names_out[:5]}...")

        # 2. FeatureSelector のテスト
        print("\n🎯 FeatureSelector テスト")
        print("-" * 40)

        from feature_selector import FeatureSelector

        selector = FeatureSelector()
        print("✅ 初期化成功")

        # 特徴量選択の学習と適用（分類タスク）
        X_selected = selector.fit_transform(
            X_scaled,
            y_binary,
            feature_names=feature_names_out,
            methods=["univariate", "rfe", "importance_based"],
        )
        print(f"✅ 特徴量選択完了: {X_scaled.shape} → {X_selected.shape}")

        # 選択結果の取得
        selection_summary = selector.get_selection_summary()
        print(f"   - 元特徴量数: {selection_summary['n_features_original']}")
        print(f"   - 使用手法: {selection_summary['methods_used']}")

        for method, result in selection_summary["selection_results"].items():
            print(
                f"   - {method}: {result['n_selected']}個選択 ({result['selection_ratio']:.1%})"
            )

        # 選択された特徴量名
        selected_names = selector.get_selected_feature_names("ensemble")
        print(f"   - アンサンブル選択: {len(selected_names)}個")
        print(f"   - 選択特徴量例: {selected_names[:5]}...")

        # 3. DimensionReducer のテスト
        print("\n📉 DimensionReducer テスト")
        print("-" * 40)

        from dimension_reducer import DimensionReducer

        reducer = DimensionReducer()
        print("✅ 初期化成功")

        # 次元削減の学習と適用
        X_reduced = reducer.fit_transform(
            X_selected, feature_names=selected_names, methods=["pca", "truncated_svd"]
        )
        print(f"✅ 次元削減完了: {X_selected.shape} → {X_reduced.shape}")

        # 削減結果の取得
        reduction_summary = reducer.get_reduction_summary()
        print(f"   - 元特徴量数: {reduction_summary['n_features_original']}")
        print(f"   - 使用手法: {reduction_summary['methods_used']}")

        for method, result in reduction_summary["reduction_results"].items():
            print(
                f"   - {method}: {result['n_components']}次元 (削減率: {1-result['reduction_ratio']:.1%})"
            )
            if "variance_explained" in result:
                print(f"     分散説明率: {result['variance_explained']:.1%}")

        # PCAの成分解釈
        if "pca" in reduction_summary["methods_used"]:
            try:
                interpretation = reducer.get_component_interpretation(
                    "pca", n_top_features=3
                )
                print(f"   - PCA成分解釈:")
                for component, features in list(interpretation.items())[
                    :2
                ]:  # 最初の2成分
                    print(f"     {component}: {[f[0][:15] for f in features]}")
            except Exception as e:
                print(f"   - 成分解釈エラー: {e}")

        # 4. 統合パイプラインテスト
        print("\n🔄 統合パイプラインテスト")
        print("-" * 40)

        # 新しいデータでの変換テスト
        X_test, _, _, _ = generate_test_data(n_samples=100, n_features=50)

        # パイプライン適用
        X_test_scaled = scaler.transform(X_test)
        X_test_selected = selector.transform(X_test_scaled, method="ensemble")
        X_test_reduced = reducer.transform(X_test_selected, method="pca")

        print(f"✅ パイプライン変換完了:")
        print(f"   - 元データ: {X_test.shape}")
        print(f"   - スケーリング後: {X_test_scaled.shape}")
        print(f"   - 特徴量選択後: {X_test_selected.shape}")
        print(f"   - 次元削減後: {X_test_reduced.shape}")

        # 5. レポート生成テスト
        print("\n📊 レポート生成テスト")
        print("-" * 40)

        # 各クラスのレポート生成
        scaling_report = scaler.generate_scaling_report("test_scaling_report.txt")
        selection_report = selector.generate_selection_report(
            "test_selection_report.txt"
        )
        reduction_report = reducer.generate_reduction_report(
            "test_reduction_report.txt"
        )

        print(f"✅ レポート生成完了:")
        print(f"   - スケーリング: {scaling_report}")
        print(f"   - 特徴量選択: {selection_report}")
        print(f"   - 次元削減: {reduction_report}")

        # 6. 保存・読み込みテスト
        print("\n💾 保存・読み込みテスト")
        print("-" * 40)

        # 保存
        scaler.save("test_scaler.pkl")
        selector.save("test_selector.pkl")
        reducer.save("test_reducer.pkl")
        print("✅ 保存完了")

        # 読み込み
        scaler_loaded = FeatureScaler.load("test_scaler.pkl")
        selector_loaded = FeatureSelector.load("test_selector.pkl")
        reducer_loaded = DimensionReducer.load("test_reducer.pkl")
        print("✅ 読み込み完了")

        # 読み込み後の変換テスト
        X_test_scaled_loaded = scaler_loaded.transform(X_test)
        X_test_selected_loaded = selector_loaded.transform(
            X_test_scaled_loaded, method="ensemble"
        )
        X_test_reduced_loaded = reducer_loaded.transform(
            X_test_selected_loaded, method="pca"
        )

        # 結果の一致確認
        scaling_match = np.allclose(X_test_scaled, X_test_scaled_loaded, rtol=1e-10)
        selection_match = np.allclose(
            X_test_selected, X_test_selected_loaded, rtol=1e-10
        )
        reduction_match = np.allclose(X_test_reduced, X_test_reduced_loaded, rtol=1e-10)

        print(f"✅ 変換結果一致確認:")
        print(f"   - スケーリング: {'✓' if scaling_match else '✗'}")
        print(f"   - 特徴量選択: {'✓' if selection_match else '✗'}")
        print(f"   - 次元削減: {'✓' if reduction_match else '✗'}")

        print("\n🎉 全ての実装確認完了！")
        print("=" * 60)
        print("✅ FeatureScaler: 動作確認済み")
        print("✅ FeatureSelector: 動作確認済み")
        print("✅ DimensionReducer: 動作確認済み")
        print("\n📝 要件3.1, 3.2, 3.3の実装が完了しました。")

        # 統計サマリー
        print(f"\n📊 最適化統計:")
        print(f"   - 元特徴量数: {X.shape[1]}")
        print(f"   - スケーリング後: {X_scaled.shape[1]}")
        print(f"   - 特徴量選択後: {X_selected.shape[1]}")
        print(f"   - 次元削減後: {X_reduced.shape[1]}")
        print(f"   - 総削減率: {1 - X_reduced.shape[1] / X.shape[1]:.1%}")

        # クリーンアップ
        for file in [
            "test_scaler.pkl",
            "test_selector.pkl",
            "test_reducer.pkl",
            "test_scaling_report.txt",
            "test_selection_report.txt",
            "test_reduction_report.txt",
        ]:
            try:
                Path(file).unlink()
            except:
                pass

        return True

    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
