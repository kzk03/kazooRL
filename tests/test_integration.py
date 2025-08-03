"""
統合テストスイート
===============

FeaturePipelineの全フロー統合テスト、エラーハンドリングとフォールバック機能のテスト、
異なる設定での動作確認テストを実装します。
"""

import json
import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import yaml

# テスト対象モジュールのパスを追加
sys.path.append(str(Path(__file__).resolve().parents[2]))

from analysis.feature_analysis import (FeatureCorrelationAnalyzer,
                                       FeatureDistributionAnalyzer,
                                       FeatureImportanceAnalyzer)
from analysis.feature_design import (DeveloperFeatureDesigner,
                                     MatchingFeatureDesigner,
                                     TaskFeatureDesigner)
from analysis.feature_optimization import (DimensionReducer, FeatureScaler,
                                           FeatureSelector)
from analysis.feature_pipeline import (ABTestConfig, AlertSeverity,
                                       FeatureABTester, FeaturePipeline,
                                       FeatureQualityMonitor,
                                       QualityMetricType, TestStatus, TestType)
from analysis.gat_optimization import (GATIntegratedOptimizer, GATInterpreter,
                                       GATOptimizer)

# ログレベルを設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegrationTestSuite(unittest.TestCase):
    """統合テストスイート"""

    @classmethod
    def setUpClass(cls):
        """テストクラス全体の前処理"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_data_dir = Path(cls.temp_dir) / "test_data"
        cls.test_data_dir.mkdir(parents=True, exist_ok=True)

        # テスト用データの生成
        cls.sample_features = np.random.randn(200, 50)
        cls.sample_labels = np.random.randint(0, 3, 200)
        cls.sample_weights = np.random.randn(50)
        cls.feature_names = [f"feature_{i}" for i in range(50)]

        logger.info(f"統合テストスイート初期化完了: temp_dir={cls.temp_dir}")

    @classmethod
    def tearDownClass(cls):
        """テストクラス全体の後処理"""
        shutil.rmtree(cls.temp_dir)
        logger.info("統合テストスイート終了")

    def setUp(self):
        """各テストの前処理"""
        self.test_output_dir = (
            Path(self.temp_dir) / f"test_output_{self._testMethodName}"
        )
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """各テストの後処理"""
        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir)


class TestFeatureAnalysisIntegration(IntegrationTestSuite):
    """特徴量分析モジュールの統合テスト"""

    def test_complete_analysis_pipeline(self):
        """完全な分析パイプラインのテスト"""
        # 重要度分析
        importance_analyzer = FeatureImportanceAnalyzer()
        importance_result = importance_analyzer.analyze_feature_importance(
            self.sample_weights, self.feature_names
        )

        self.assertIsInstance(importance_result, dict)
        self.assertIn("importance_ranking", importance_result)
        self.assertIn("category_comparison", importance_result)

        # 相関分析
        correlation_analyzer = FeatureCorrelationAnalyzer()
        correlation_result = correlation_analyzer.analyze_feature_correlations(
            self.sample_features
        )

        self.assertIsInstance(correlation_result, dict)
        self.assertIn("correlation_matrix", correlation_result)
        self.assertIn("high_correlation_pairs", correlation_result)

        # 分布分析
        distribution_analyzer = FeatureDistributionAnalyzer()
        distribution_result = distribution_analyzer.analyze_feature_distributions(
            self.sample_features
        )

        self.assertIsInstance(distribution_result, dict)
        self.assertIn("distribution_stats", distribution_result)
        self.assertIn("normality_tests", distribution_result)

        logger.info("特徴量分析パイプライン統合テスト完了")

    def test_analysis_error_handling(self):
        """分析モジュールのエラーハンドリングテスト"""
        importance_analyzer = FeatureImportanceAnalyzer()

        # 空のデータでのエラーハンドリング
        with self.assertRaises((ValueError, IndexError)):
            importance_analyzer.analyze_feature_importance([], [])

        # 次元不一致でのエラーハンドリング
        with self.assertRaises((ValueError, IndexError)):
            importance_analyzer.analyze_feature_importance(
                np.random.randn(10), [f"feature_{i}" for i in range(5)]
            )

        logger.info("分析モジュールエラーハンドリングテスト完了")


class TestFeatureDesignIntegration(IntegrationTestSuite):
    """特徴量設計モジュールの統合テスト"""

    def test_complete_design_pipeline(self):
        """完全な設計パイプラインのテスト"""
        # サンプルデータ準備
        task_data = {
            "task_id": [1, 2, 3],
            "title": ["Fix bug", "Add feature", "Update docs"],
            "description": [
                "Fix critical bug",
                "Add new API feature",
                "Update documentation",
            ],
            "priority": ["high", "medium", "low"],
        }

        developer_data = {
            "developer_id": [1, 2, 3],
            "commits_count": [100, 50, 200],
            "expertise_languages": [
                ["Python", "JavaScript"],
                ["Java"],
                ["Python", "Go"],
            ],
            "recent_activity": [10, 5, 15],
        }

        # タスク特徴量設計
        task_designer = TaskFeatureDesigner()
        task_features = task_designer.design_enhanced_task_features(task_data)

        self.assertIsInstance(task_features, dict)
        self.assertIn("urgency_features", task_features)
        self.assertIn("complexity_features", task_features)

        # 開発者特徴量設計
        developer_designer = DeveloperFeatureDesigner()
        developer_features = developer_designer.design_enhanced_developer_features(
            developer_data
        )

        self.assertIsInstance(developer_features, dict)
        self.assertIn("expertise_features", developer_features)
        self.assertIn("activity_pattern_features", developer_features)

        # マッチング特徴量設計
        matching_designer = MatchingFeatureDesigner()
        matching_features = matching_designer.design_enhanced_matching_features(
            task_data, developer_data
        )

        self.assertIsInstance(matching_features, dict)
        self.assertIn("temporal_proximity_features", matching_features)
        self.assertIn("technical_compatibility_features", matching_features)

        logger.info("特徴量設計パイプライン統合テスト完了")

    def test_design_with_missing_data(self):
        """欠損データでの設計テスト"""
        # 不完全なデータ
        incomplete_task_data = {
            "task_id": [1, 2],
            "title": ["Fix bug", None],
            "description": [None, "Add feature"],
        }

        task_designer = TaskFeatureDesigner()

        # 欠損データでも処理が完了することを確認
        try:
            task_features = task_designer.design_enhanced_task_features(
                incomplete_task_data
            )
            self.assertIsInstance(task_features, dict)
        except Exception as e:
            self.fail(f"欠損データ処理でエラーが発生: {e}")

        logger.info("欠損データ処理テスト完了")


class TestFeatureOptimizationIntegration(IntegrationTestSuite):
    """特徴量最適化モジュールの統合テスト"""

    def test_complete_optimization_pipeline(self):
        """完全な最適化パイプラインのテスト"""
        # スケーリング
        scaler = FeatureScaler()
        scaled_features = scaler.fit_transform(self.sample_features)

        self.assertEqual(scaled_features.shape, self.sample_features.shape)
        self.assertFalse(np.array_equal(scaled_features, self.sample_features))

        # 特徴量選択
        selector = FeatureSelector()
        selected_features, selected_indices = selector.select_features(
            scaled_features, self.sample_labels, method="univariate", k=20
        )

        self.assertEqual(selected_features.shape[1], 20)
        self.assertEqual(len(selected_indices), 20)

        # 次元削減
        reducer = DimensionReducer()
        reduced_features = reducer.reduce_dimensions(
            selected_features, method="pca", n_components=10
        )

        self.assertEqual(reduced_features.shape[1], 10)
        self.assertEqual(reduced_features.shape[0], self.sample_features.shape[0])

        logger.info("特徴量最適化パイプライン統合テスト完了")

    def test_optimization_consistency(self):
        """最適化の一貫性テスト"""
        scaler = FeatureScaler()

        # 同じデータに対して同じ結果が得られることを確認
        scaled1 = scaler.fit_transform(self.sample_features)
        scaled2 = scaler.transform(self.sample_features)

        np.testing.assert_array_almost_equal(scaled1, scaled2, decimal=10)

        logger.info("最適化一貫性テスト完了")


class TestGATOptimizationIntegration(IntegrationTestSuite):
    """GAT最適化モジュールの統合テスト"""

    def test_gat_optimization_pipeline(self):
        """GAT最適化パイプラインのテスト"""
        # モックGATモデルを作成
        mock_gat_model = Mock()
        mock_gat_model.get_attention_weights.return_value = np.random.randn(10, 10, 8)
        mock_gat_model.get_embeddings.return_value = np.random.randn(200, 64)

        # GAT最適化
        gat_optimizer = GATOptimizer()

        # 最適次元数分析（モック使用）
        with patch.object(
            gat_optimizer, "_evaluate_embedding_quality", return_value=0.85
        ):
            optimal_dims = gat_optimizer.find_optimal_dimensions(
                mock_gat_model, range(16, 65, 16)
            )
            self.assertIsInstance(optimal_dims, dict)
            self.assertIn("optimal_dimensions", optimal_dims)

        # GAT解釈
        gat_interpreter = GATInterpreter()

        # 次元解釈（モック使用）
        interpretation_result = gat_interpreter.interpret_gat_dimensions(
            mock_gat_model.get_embeddings(), self.feature_names[:10]
        )

        self.assertIsInstance(interpretation_result, dict)
        self.assertIn("dimension_interpretations", interpretation_result)

        # 統合最適化
        integrated_optimizer = GATIntegratedOptimizer()

        # 特徴量組み合わせ検索（モック使用）
        with patch.object(
            integrated_optimizer, "_evaluate_feature_combination", return_value=0.90
        ):
            combination_result = (
                integrated_optimizer.search_optimal_feature_combinations(
                    self.sample_features,
                    mock_gat_model.get_embeddings(),
                    self.sample_labels,
                )
            )

            self.assertIsInstance(combination_result, dict)
            self.assertIn("optimal_combination", combination_result)

        logger.info("GAT最適化パイプライン統合テスト完了")


class TestFeaturePipelineIntegration(IntegrationTestSuite):
    """特徴量パイプライン統合テスト"""

    def test_full_pipeline_execution(self):
        """完全パイプライン実行テスト"""
        # テスト用設定ファイル作成
        config = {
            "pipeline": {
                "stages": ["analysis", "design", "optimization", "evaluation"],
                "enable_cache": True,
                "error_handling": "continue",
            },
            "analysis": {
                "importance_analysis": True,
                "correlation_analysis": True,
                "distribution_analysis": True,
            },
            "design": {
                "task_features": True,
                "developer_features": True,
                "matching_features": True,
            },
            "optimization": {
                "scaling": True,
                "selection": True,
                "dimension_reduction": True,
            },
            "evaluation": {"quality_metrics": True, "performance_comparison": True},
        }

        config_file = self.test_output_dir / "pipeline_config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        # パイプライン実行
        pipeline = FeaturePipeline(
            config_path=str(config_file), cache_dir=str(self.test_output_dir / "cache")
        )

        pipeline.initialize_components()

        # テストデータでパイプライン実行
        test_data = {
            "features": self.sample_features,
            "labels": self.sample_labels,
            "weights": self.sample_weights,
        }

        result = pipeline.run_full_pipeline(test_data)

        self.assertIsInstance(result, dict)
        self.assertIn("pipeline_id", result)
        self.assertIn("stage_results", result)
        self.assertIn("performance_summary", result)
        self.assertTrue(len(result["stages_executed"]) > 0)

        logger.info("完全パイプライン実行テスト完了")

    def test_pipeline_error_handling(self):
        """パイプラインエラーハンドリングテスト"""
        # エラーが発生する設定
        config = {
            "pipeline": {
                "stages": ["analysis", "invalid_stage"],
                "error_handling": "continue",
            }
        }

        config_file = self.test_output_dir / "error_config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        pipeline = FeaturePipeline(
            config_path=str(config_file), cache_dir=str(self.test_output_dir / "cache")
        )

        pipeline.initialize_components()

        test_data = {"features": self.sample_features, "labels": self.sample_labels}

        # エラーハンドリングモードで実行
        result = pipeline.run_full_pipeline(test_data)

        self.assertIsInstance(result, dict)
        self.assertIn("error", result)  # エラー情報が含まれる

        logger.info("パイプラインエラーハンドリングテスト完了")

    def test_pipeline_caching(self):
        """パイプラインキャッシュ機能テスト"""
        config = {
            "pipeline": {"stages": ["analysis"], "enable_cache": True},
            "analysis": {"importance_analysis": True},
        }

        config_file = self.test_output_dir / "cache_config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        pipeline = FeaturePipeline(
            config_path=str(config_file), cache_dir=str(self.test_output_dir / "cache")
        )

        pipeline.initialize_components()

        test_data = {"features": self.sample_features, "weights": self.sample_weights}

        # 1回目実行
        result1 = pipeline.run_full_pipeline(test_data)

        # 2回目実行（キャッシュから読み込まれるはず）
        result2 = pipeline.run_full_pipeline(test_data)

        self.assertEqual(result1["pipeline_id"], result2["pipeline_id"])

        logger.info("パイプラインキャッシュ機能テスト完了")


class TestQualityMonitoringIntegration(IntegrationTestSuite):
    """品質監視統合テスト"""

    def test_quality_monitoring_workflow(self):
        """品質監視ワークフローテスト"""
        monitor = FeatureQualityMonitor(output_dir=str(self.test_output_dir))

        # 監視開始
        monitor.start_monitoring(
            self.sample_features, self.sample_labels, self.feature_names
        )

        self.assertTrue(monitor.is_monitoring)
        self.assertTrue(len(monitor.baseline_metrics) > 0)

        # 現在品質監視
        modified_features = (
            self.sample_features + np.random.randn(*self.sample_features.shape) * 0.1
        )

        monitoring_result = monitor.monitor_current_quality(
            modified_features, self.sample_labels, self.feature_names
        )

        self.assertIsInstance(monitoring_result, dict)
        self.assertIn("current_metrics", monitoring_result)
        self.assertIn("quality_changes", monitoring_result)
        self.assertIn("overall_quality_score", monitoring_result)

        # レポート生成
        report_path = monitor.generate_quality_report(
            include_visualizations=False, report_format="json"
        )

        self.assertTrue(Path(report_path).exists())

        # 監視停止
        final_summary = monitor.stop_monitoring()

        self.assertFalse(monitor.is_monitoring)
        self.assertIsInstance(final_summary, dict)

        logger.info("品質監視ワークフローテスト完了")


class TestABTestingIntegration(IntegrationTestSuite):
    """A/Bテスト統合テスト"""

    def test_ab_testing_workflow(self):
        """A/Bテストワークフローテスト"""
        ab_tester = FeatureABTester(output_dir=str(self.test_output_dir))

        # 特徴量セット登録
        baseline_features = self.sample_features
        treatment_features = (
            self.sample_features + np.random.randn(*self.sample_features.shape) * 0.1
        )

        ab_tester.register_feature_set(
            "baseline_v1",
            baseline_features,
            self.sample_labels,
            self.feature_names,
            "ベースライン特徴量セット",
        )

        ab_tester.register_feature_set(
            "treatment_v1",
            treatment_features,
            self.sample_labels,
            self.feature_names,
            "改善版特徴量セット",
        )

        # A/Bテスト設定
        test_config = ABTestConfig(
            test_id="test_001",
            test_name="特徴量改善効果テスト",
            test_type=TestType.PERFORMANCE_COMPARISON,
            baseline_features="baseline_v1",
            treatment_features="treatment_v1",
            success_metrics=["accuracy", "f1_score"],
            minimum_sample_size=50,
            significance_level=0.05,
        )

        # テスト作成と実行
        test_id = ab_tester.create_ab_test(test_config)

        # モックモデルを使用してテスト実行
        mock_models = {"random_forest": Mock(), "logistic_regression": Mock()}

        with patch(
            "sklearn.model_selection.cross_val_score",
            return_value=np.array([0.8, 0.85, 0.82, 0.88, 0.86]),
        ):
            test_result = ab_tester.run_ab_test(test_id, mock_models)

        self.assertEqual(test_result.test_id, test_id)
        self.assertEqual(test_result.status, TestStatus.COMPLETED)
        self.assertIsInstance(test_result.baseline_metrics, dict)
        self.assertIsInstance(test_result.treatment_metrics, dict)
        self.assertIsInstance(test_result.statistical_results, dict)

        # レポート生成
        report_path = ab_tester.generate_comparison_report(
            test_id, include_visualizations=False, report_format="json"
        )

        self.assertTrue(Path(report_path).exists())

        logger.info("A/Bテストワークフローテスト完了")


class TestEndToEndIntegration(IntegrationTestSuite):
    """エンドツーエンド統合テスト"""

    def test_complete_feature_engineering_workflow(self):
        """完全な特徴量エンジニアリングワークフローテスト"""
        # 1. パイプライン設定
        config = {
            "pipeline": {
                "stages": ["analysis", "design", "optimization", "evaluation"],
                "enable_cache": True,
            },
            "analysis": {"importance_analysis": True, "correlation_analysis": True},
            "design": {"task_features": True, "developer_features": True},
            "optimization": {"scaling": True, "selection": True},
            "evaluation": {"quality_metrics": True},
        }

        config_file = self.test_output_dir / "e2e_config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        # 2. パイプライン実行
        pipeline = FeaturePipeline(
            config_path=str(config_file),
            cache_dir=str(self.test_output_dir / "pipeline_cache"),
        )

        pipeline.initialize_components()

        test_data = {"features": self.sample_features, "weights": self.sample_weights}
        pipeline_result = pipeline.run_full_pipeline(test_data)

        # 3. 品質監視
        monitor = FeatureQualityMonitor(
            output_dir=str(self.test_output_dir / "monitoring")
        )
        monitor.start_monitoring(
            self.sample_features, self.sample_labels, self.feature_names
        )

        monitoring_result = monitor.monitor_current_quality(
            self.sample_features, self.sample_labels, self.feature_names
        )

        # 4. A/Bテスト
        ab_tester = FeatureABTester(output_dir=str(self.test_output_dir / "ab_testing"))

        ab_tester.register_feature_set(
            "original", self.sample_features, self.sample_labels
        )
        ab_tester.register_feature_set(
            "optimized", self.sample_features * 1.1, self.sample_labels
        )

        test_config = ABTestConfig(
            test_id="e2e_test",
            test_name="E2Eテスト",
            test_type=TestType.PERFORMANCE_COMPARISON,
            baseline_features="original",
            treatment_features="optimized",
            success_metrics=["accuracy"],
        )

        test_id = ab_tester.create_ab_test(test_config)

        with patch(
            "sklearn.model_selection.cross_val_score",
            return_value=np.array([0.8, 0.85, 0.82]),
        ):
            ab_result = ab_tester.run_ab_test(test_id)

        # 結果検証
        self.assertIsInstance(pipeline_result, dict)
        self.assertIn("pipeline_id", pipeline_result)

        self.assertIsInstance(monitoring_result, dict)
        self.assertIn("overall_quality_score", monitoring_result)

        self.assertEqual(ab_result.status, TestStatus.COMPLETED)

        logger.info("エンドツーエンド統合テスト完了")


class TestConfigurationValidation(IntegrationTestSuite):
    """設定検証テスト"""

    def test_various_pipeline_configurations(self):
        """様々なパイプライン設定のテスト"""
        # 最小設定
        minimal_config = {
            "pipeline": {"stages": ["analysis"]},
            "analysis": {"importance_analysis": True},
        }

        # 最大設定
        maximal_config = {
            "pipeline": {
                "stages": [
                    "analysis",
                    "design",
                    "optimization",
                    "gat_enhancement",
                    "evaluation",
                ],
                "enable_cache": True,
                "parallel_execution": False,
                "error_handling": "continue",
            },
            "analysis": {
                "importance_analysis": True,
                "correlation_analysis": True,
                "distribution_analysis": True,
            },
            "design": {
                "task_features": True,
                "developer_features": True,
                "matching_features": True,
            },
            "optimization": {
                "scaling": True,
                "selection": True,
                "dimension_reduction": True,
            },
            "gat_enhancement": {
                "optimization": True,
                "interpretation": True,
                "integration": True,
            },
            "evaluation": {"quality_metrics": True, "performance_comparison": True},
        }

        configs = [minimal_config, maximal_config]

        for i, config in enumerate(configs):
            config_file = self.test_output_dir / f"config_{i}.yaml"
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(config, f)

            pipeline = FeaturePipeline(
                config_path=str(config_file),
                cache_dir=str(self.test_output_dir / f"cache_{i}"),
            )

            pipeline.initialize_components()
            self.assertTrue(pipeline.is_initialized)

            # 基本実行テスト
            test_data = {
                "features": self.sample_features[:50],
                "weights": self.sample_weights,
            }

            try:
                result = pipeline.run_full_pipeline(test_data)
                self.assertIsInstance(result, dict)
                logger.info(f"設定 {i} の実行完了")
            except Exception as e:
                self.fail(f"設定 {i} の実行でエラー: {e}")

        logger.info("設定検証テスト完了")


def run_integration_tests():
    """統合テストの実行"""
    # テストスイートを作成
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 各テストクラスをスイートに追加
    test_classes = [
        TestFeatureAnalysisIntegration,
        TestFeatureDesignIntegration,
        TestFeatureOptimizationIntegration,
        TestGATOptimizationIntegration,
        TestFeaturePipelineIntegration,
        TestQualityMonitoringIntegration,
        TestABTestingIntegration,
        TestEndToEndIntegration,
        TestConfigurationValidation,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # テストランナーで実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 結果サマリー
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = (
        (total_tests - failures - errors) / total_tests * 100 if total_tests > 0 else 0
    )

    print(f"\n" + "=" * 60)
    print("統合テストスイート実行結果")
    print("=" * 60)
    print(f"総テスト数: {total_tests}")
    print(f"成功: {total_tests - failures - errors}")
    print(f"失敗: {failures}")
    print(f"エラー: {errors}")
    print(f"成功率: {success_rate:.1f}%")
    print("=" * 60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
