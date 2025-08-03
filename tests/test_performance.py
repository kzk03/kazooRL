"""
パフォーマンステストスイート
==========================

特徴量抽出速度のベンチマークテスト、メモリ使用量とスケーラビリティのテスト、
大規模データセットでの動作確認テストを実装します。
"""

import gc
import logging
import os
import shutil
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import yaml

# テスト対象モジュールのパスを追加
sys.path.append(str(Path(__file__).resolve().parents[2]))

from analysis.feature_analysis import (
    FeatureCorrelationAnalyzer,
    FeatureDistributionAnalyzer,
    FeatureImportanceAnalyzer,
)
from analysis.feature_design import (
    DeveloperFeatureDesigner,
    MatchingFeatureDesigner,
    TaskFeatureDesigner,
)
from analysis.feature_optimization import (
    DimensionReducer,
    FeatureScaler,
    FeatureSelector,
)
from analysis.feature_pipeline import FeaturePipeline

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_percent: float
    throughput: float  # samples/second


@dataclass
class ScalabilityResult:
    """スケーラビリティ結果"""
    data_sizes: List[int]
    execution_times: List[float]
    memory_usages: List[float]
    throughputs: List[float]
    scaling_factor: float  # 理想的な線形スケーリングからの乖離


class PerformanceTestSuite(unittest.TestCase):
    """パフォーマンステストスイート"""
    
    @classmethod
    def setUpClass(cls):
        """テストクラス全体の前処理"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_data_dir = Path(cls.temp_dir) / 'perf_test_data'
        cls.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # ベンチマーク基準値
        cls.benchmark_thresholds = {
            'small_dataset_time': 1.0,     # 小規模データセット処理時間(秒)
            'medium_dataset_time': 10.0,   # 中規模データセット処理時間(秒)
            'large_dataset_time': 60.0,    # 大規模データセット処理時間(秒)
            'memory_efficiency': 2.0,      # メモリ効率(データサイズに対する倍率)
            'min_throughput': 1000,        # 最小スループット(samples/second)
        }
        
        logger.info(f"パフォーマンステストスイート初期化完了: temp_dir={cls.temp_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """テストクラス全体の後処理"""
        shutil.rmtree(cls.temp_dir)
        logger.info("パフォーマンステストスイート終了")
    
    def setUp(self):
        """各テストの前処理"""
        gc.collect()  # ガベージコレクション実行
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def tearDown(self):
        """各テストの後処理"""
        gc.collect()
    
    def _measure_performance(self, func, *args, **kwargs) -> PerformanceMetrics:
        """関数のパフォーマンスを測定"""
        # 初期状態記録
        start_memory = self.process.memory_info().rss / 1024 / 1024
        start_time = time.time()
        start_cpu = self.process.cpu_percent(interval=None)
        
        # 関数実行
        result = func(*args, **kwargs)
        
        # 終了状態記録
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        end_cpu = self.process.cpu_percent(interval=None)
        
        # ピークメモリ使用量測定（簡略化）
        peak_memory = max(start_memory, end_memory)
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        cpu_percent = end_cpu
        
        # スループット計算（サンプル数ベース）
        if hasattr(result, '__len__') and len(args) > 0:
            if hasattr(args[0], 'shape'):
                sample_count = args[0].shape[0]
            elif hasattr(args[0], '__len__'):
                sample_count = len(args[0])
            else:
                sample_count = 1000  # デフォルト値
        else:
            sample_count = 1000
            
        throughput = sample_count / execution_time if execution_time > 0 else 0
        
        return PerformanceMetrics(
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            peak_memory_mb=peak_memory,
            cpu_percent=cpu_percent,
            throughput=throughput
        )
    
    def _generate_test_data(self, n_samples: int, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
        """テストデータ生成"""
        np.random.seed(42)  # 再現性のため
        features = np.random.randn(n_samples, n_features)
        labels = np.random.randint(0, 3, n_samples)
        return features, labels


class TestFeatureAnalysisPerformance(PerformanceTestSuite):
    """特徴量分析パフォーマンステスト"""
    
    def test_importance_analysis_scalability(self):
        """重要度分析のスケーラビリティテスト"""
        data_sizes = [100, 500, 1000, 5000]
        feature_counts = [50, 100, 200, 500]
        
        analyzer = FeatureImportanceAnalyzer()
        results = []
        
        for n_features in feature_counts:
            weights = np.random.randn(n_features)
            feature_names = [f'feature_{i}' for i in range(n_features)]
            
            metrics = self._measure_performance(
                analyzer.analyze_feature_importance,
                weights, feature_names
            )
            
            results.append(metrics)
            
            # パフォーマンス閾値チェック
            self.assertLess(metrics.execution_time, 5.0, 
                          f"重要度分析が遅すぎます: {metrics.execution_time:.2f}s > 5.0s")
            self.assertLess(metrics.memory_usage_mb, 100, 
                          f"メモリ使用量が多すぎます: {metrics.memory_usage_mb:.1f}MB > 100MB")
        
        logger.info(f"重要度分析スケーラビリティテスト完了: {len(results)}パターン")
    
    def test_correlation_analysis_performance(self):
        """相関分析パフォーマンステスト"""
        test_cases = [
            (1000, 50),   # 小規模
            (5000, 100),  # 中規模  
            (10000, 200), # 大規模
        ]
        
        analyzer = FeatureCorrelationAnalyzer()
        
        for n_samples, n_features in test_cases:
            features, _ = self._generate_test_data(n_samples, n_features)
            
            metrics = self._measure_performance(
                analyzer.analyze_feature_correlations,
                features
            )
            
            # データサイズに応じた閾値チェック
            expected_time = n_features * n_features * 1e-6  # O(n²)の想定
            self.assertLess(metrics.execution_time, expected_time * 1000,
                          f"相関分析が期待より遅いです: {metrics.execution_time:.3f}s")
            
            logger.info(f"相関分析 ({n_samples}×{n_features}): "
                       f"時間={metrics.execution_time:.3f}s, "
                       f"メモリ={metrics.memory_usage_mb:.1f}MB")
    
    def test_distribution_analysis_throughput(self):
        """分布分析スループットテスト"""
        analyzer = FeatureDistributionAnalyzer()
        
        # 様々なサイズでテスト
        test_sizes = [(1000, 50), (5000, 100), (10000, 200)]
        
        for n_samples, n_features in test_sizes:
            features, _ = self._generate_test_data(n_samples, n_features)
            
            metrics = self._measure_performance(
                analyzer.analyze_feature_distributions,
                features
            )
            
            # スループット要件チェック
            min_throughput = self.benchmark_thresholds['min_throughput']
            self.assertGreater(metrics.throughput, min_throughput,
                             f"スループットが低すぎます: {metrics.throughput:.1f} < {min_throughput}")
            
            logger.info(f"分布分析スループット ({n_samples}×{n_features}): "
                       f"{metrics.throughput:.1f} samples/sec")


class TestFeatureDesignPerformance(PerformanceTestSuite):
    """特徴量設計パフォーマンステスト"""
    
    def test_task_feature_design_performance(self):
        """タスク特徴量設計パフォーマンステスト"""
        designer = TaskFeatureDesigner()
        
        # 様々なサイズのタスクデータを生成
        test_sizes = [100, 1000, 5000, 10000]
        
        for n_tasks in test_sizes:
            task_data = {
                'task_id': list(range(n_tasks)),
                'title': [f'Task {i}' for i in range(n_tasks)],
                'description': [f'Description for task {i}' * 10 for i in range(n_tasks)],
                'priority': np.random.choice(['low', 'medium', 'high'], n_tasks).tolist()
            }
            
            metrics = self._measure_performance(
                designer.design_enhanced_task_features,
                task_data
            )
            
            # 線形スケーリング要件チェック
            expected_time = n_tasks * 1e-4  # 線形スケーリング想定
            self.assertLess(metrics.execution_time, expected_time * 10,
                          f"タスク特徴量設計がスケールしません: {metrics.execution_time:.3f}s")
            
            logger.info(f"タスク特徴量設計 ({n_tasks}タスク): "
                       f"時間={metrics.execution_time:.3f}s, "
                       f"スループット={metrics.throughput:.1f}/sec")
    
    def test_developer_feature_design_memory_usage(self):
        """開発者特徴量設計メモリ使用量テスト"""
        designer = DeveloperFeatureDesigner()
        
        # 大規模データでメモリ使用量テスト
        n_developers = 10000
        developer_data = {
            'developer_id': list(range(n_developers)),
            'commits_count': np.random.randint(1, 1000, n_developers).tolist(),
            'expertise_languages': [
                np.random.choice(['Python', 'JavaScript', 'Java', 'Go', 'Rust'], 
                               np.random.randint(1, 4)).tolist() 
                for _ in range(n_developers)
            ],
            'recent_activity': np.random.randint(0, 100, n_developers).tolist()
        }
        
        metrics = self._measure_performance(
            designer.design_enhanced_developer_features,
            developer_data
        )
        
        # メモリ効率要件チェック
        data_size_mb = sys.getsizeof(developer_data) / 1024 / 1024
        memory_ratio = metrics.memory_usage_mb / data_size_mb
        
        max_memory_ratio = self.benchmark_thresholds['memory_efficiency']
        self.assertLess(memory_ratio, max_memory_ratio,
                       f"メモリ効率が悪いです: {memory_ratio:.1f}x > {max_memory_ratio}x")
        
        logger.info(f"開発者特徴量設計メモリ効率: {memory_ratio:.1f}x")
    
    def test_matching_feature_design_complexity(self):
        """マッチング特徴量設計複雑度テスト"""
        designer = MatchingFeatureDesigner()
        
        # 複雑度テスト用データ
        complexity_levels = [
            (100, 50),    # 低複雑度
            (500, 200),   # 中複雑度
            (1000, 500),  # 高複雑度
        ]
        
        for n_tasks, n_developers in complexity_levels:
            task_data = {
                'task_id': list(range(n_tasks)),
                'title': [f'Task {i}' for i in range(n_tasks)],
                'description': [f'Description {i}' for i in range(n_tasks)]
            }
            
            developer_data = {
                'developer_id': list(range(n_developers)),
                'commits_count': np.random.randint(1, 100, n_developers).tolist(),
                'expertise_languages': [['Python'] for _ in range(n_developers)]
            }
            
            metrics = self._measure_performance(
                designer.design_enhanced_matching_features,
                task_data, developer_data
            )
            
            # 複雑度チェック（O(n*m)想定）
            expected_complexity = n_tasks * n_developers * 1e-7
            self.assertLess(metrics.execution_time, expected_complexity * 100,
                          f"マッチング特徴量設計の複雑度が高すぎます: {metrics.execution_time:.3f}s")
            
            logger.info(f"マッチング特徴量設計 ({n_tasks}×{n_developers}): "
                       f"時間={metrics.execution_time:.3f}s")


class TestFeatureOptimizationPerformance(PerformanceTestSuite):
    """特徴量最適化パフォーマンステスト"""
    
    def test_scaling_performance_benchmark(self):
        """スケーリングパフォーマンスベンチマーク"""
        scaler = FeatureScaler()
        
        # ベンチマークデータサイズ
        benchmark_sizes = [
            (1000, 100),    # 小規模
            (10000, 200),   # 中規模
            (50000, 500),   # 大規模
        ]
        
        scaling_methods = ['standard', 'minmax', 'robust']
        
        for method in scaling_methods:
            method_results = []
            
            for n_samples, n_features in benchmark_sizes:
                features, _ = self._generate_test_data(n_samples, n_features)
                
                metrics = self._measure_performance(
                    scaler.fit_transform,
                    features, method=method
                )
                
                method_results.append(metrics)
                
                # 基準時間チェック
                size_category = 'small' if n_samples <= 1000 else 'medium' if n_samples <= 10000 else 'large'
                threshold_key = f'{size_category}_dataset_time'
                
                self.assertLess(metrics.execution_time, self.benchmark_thresholds[threshold_key],
                              f"{method}スケーリングが遅すぎます: {metrics.execution_time:.2f}s")
            
            logger.info(f"{method}スケーリングベンチマーク完了: {len(method_results)}サイズ")
    
    def test_feature_selection_scalability(self):
        """特徴量選択スケーラビリティテスト"""
        selector = FeatureSelector()
        
        # スケーラビリティテスト
        feature_counts = [50, 100, 200, 500, 1000]
        scalability_results = []
        
        for n_features in feature_counts:
            features, labels = self._generate_test_data(5000, n_features)
            
            metrics = self._measure_performance(
                selector.select_features,
                features, labels, method='univariate', k=min(20, n_features//2)
            )
            
            scalability_results.append(metrics)
        
        # スケーラビリティ分析
        execution_times = [m.execution_time for m in scalability_results]
        
        # 理想的な線形スケーリングからの乖離を計算
        if len(execution_times) > 1:
            scaling_factor = execution_times[-1] / execution_times[0]
            feature_ratio = feature_counts[-1] / feature_counts[0]
            
            # 線形よりも悪いスケーリングでないことを確認
            self.assertLess(scaling_factor, feature_ratio * 2,
                          f"特徴量選択のスケーラビリティが悪いです: {scaling_factor:.1f}x")
        
        logger.info(f"特徴量選択スケーラビリティテスト完了: {len(scalability_results)}ポイント")
    
    def test_dimension_reduction_memory_efficiency(self):
        """次元削減メモリ効率テスト"""
        reducer = DimensionReducer()
        
        # メモリ効率テスト用大規模データ
        n_samples, n_features = 20000, 1000
        features, _ = self._generate_test_data(n_samples, n_features)
        
        reduction_methods = ['pca', 'umap']
        target_dimensions = [50, 100, 200]
        
        for method in reduction_methods:
            for n_components in target_dimensions:
                if method == 'umap' and n_components > 100:
                    continue  # UMAPは高次元で重い
                
                metrics = self._measure_performance(
                    reducer.reduce_dimensions,
                    features, method=method, n_components=n_components
                )
                
                # メモリ効率チェック
                input_size_mb = features.nbytes / 1024 / 1024
                memory_ratio = metrics.memory_usage_mb / input_size_mb
                
                self.assertLess(memory_ratio, 3.0,
                              f"{method}次元削減のメモリ使用量が多すぎます: {memory_ratio:.1f}x")
                
                logger.info(f"{method}次元削減 (→{n_components}): "
                           f"時間={metrics.execution_time:.2f}s, "
                           f"メモリ効率={memory_ratio:.1f}x")


class TestPipelinePerformance(PerformanceTestSuite):
    """パイプラインパフォーマンステスト"""
    
    def test_full_pipeline_benchmark(self):
        """完全パイプラインベンチマーク"""
        # ベンチマーク設定
        config = {
            'pipeline': {
                'stages': ['analysis', 'design', 'optimization'],
                'enable_cache': False,  # キャッシュを無効にして純粋な性能測定
                'error_handling': 'continue'
            },
            'analysis': {
                'importance_analysis': True,
                'correlation_analysis': True,
                'distribution_analysis': True
            },
            'design': {
                'task_features': True,
                'developer_features': True
            },
            'optimization': {
                'scaling': True,
                'selection': True,
                'dimension_reduction': True
            }
        }
        
        config_file = Path(self.temp_dir) / 'benchmark_config.yaml'
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        pipeline = FeaturePipeline(
            config_path=str(config_file),
            cache_dir=str(Path(self.temp_dir) / 'benchmark_cache')
        )
        
        pipeline.initialize_components()
        
        # 様々なサイズでベンチマーク
        benchmark_data_sizes = [
            (1000, 50),   # 小規模
            (5000, 100),  # 中規模
            (10000, 200), # 大規模
        ]
        
        for n_samples, n_features in benchmark_data_sizes:
            features, labels = self._generate_test_data(n_samples, n_features)
            weights = np.random.randn(n_features)
            
            test_data = {
                'features': features,
                'labels': labels,
                'weights': weights
            }
            
            metrics = self._measure_performance(
                pipeline.run_full_pipeline,
                test_data
            )
            
            # データサイズに応じた性能要件チェック
            size_category = 'small' if n_samples <= 1000 else 'medium' if n_samples <= 5000 else 'large'
            threshold_key = f'{size_category}_dataset_time'
            
            self.assertLess(metrics.execution_time, self.benchmark_thresholds[threshold_key],
                          f"パイプライン実行が遅すぎます ({n_samples}×{n_features}): "
                          f"{metrics.execution_time:.2f}s > {self.benchmark_thresholds[threshold_key]}s")
            
            logger.info(f"パイプラインベンチマーク ({n_samples}×{n_features}): "
                       f"時間={metrics.execution_time:.2f}s, "
                       f"メモリ={metrics.memory_usage_mb:.1f}MB, "
                       f"スループット={metrics.throughput:.1f}/sec")
    
    def test_pipeline_caching_performance(self):
        """パイプラインキャッシュ性能テスト"""
        config = {
            'pipeline': {
                'stages': ['analysis', 'optimization'],
                'enable_cache': True
            },
            'analysis': {'importance_analysis': True},
            'optimization': {'scaling': True}
        }
        
        config_file = Path(self.temp_dir) / 'cache_perf_config.yaml'
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        pipeline = FeaturePipeline(
            config_path=str(config_file),
            cache_dir=str(Path(self.temp_dir) / 'cache_perf')
        )
        
        pipeline.initialize_components()
        
        features, labels = self._generate_test_data(5000, 100)
        test_data = {
            'features': features,
            'labels': labels,
            'weights': np.random.randn(100)
        }
        
        # 初回実行（キャッシュなし）
        metrics_first = self._measure_performance(
            pipeline.run_full_pipeline,
            test_data
        )
        
        # 2回目実行（キャッシュあり）
        metrics_cached = self._measure_performance(
            pipeline.run_full_pipeline,
            test_data
        )
        
        # キャッシュ効果の確認
        speedup_ratio = metrics_first.execution_time / metrics_cached.execution_time
        
        self.assertGreater(speedup_ratio, 2.0,
                         f"キャッシュ効果が不十分です: {speedup_ratio:.1f}x speedup")
        
        logger.info(f"キャッシュ性能: {speedup_ratio:.1f}x高速化 "
                   f"({metrics_first.execution_time:.3f}s → {metrics_cached.execution_time:.3f}s)")


class TestLargeScaleDatasetPerformance(PerformanceTestSuite):
    """大規模データセットパフォーマンステスト"""
    
    def test_extreme_large_dataset_handling(self):
        """極大データセット処理テスト"""
        # メモリ制限を考慮した大規模テスト
        large_sizes = [
            (100000, 100),   # 10万サンプル
            (50000, 500),    # 5万サンプル×500特徴量
        ]
        
        for n_samples, n_features in large_sizes:
            logger.info(f"大規模データセットテスト開始: {n_samples}×{n_features}")
            
            # メモリ使用量を監視しながらデータ生成
            initial_memory = self.process.memory_info().rss / 1024 / 1024
            
            features, labels = self._generate_test_data(n_samples, n_features)
            
            data_memory = self.process.memory_info().rss / 1024 / 1024 - initial_memory
            
            # データサイズが妥当な範囲内であることを確認
            expected_data_size = n_samples * n_features * 8 / 1024 / 1024  # float64で計算
            
            self.assertLess(data_memory, expected_data_size * 3,
                          f"データ生成時のメモリ使用量が想定より多い: {data_memory:.1f}MB")
            
            # 基本的な処理が実行可能であることを確認
            analyzer = FeatureImportanceAnalyzer()
            weights = np.random.randn(n_features)
            feature_names = [f'f_{i}' for i in range(n_features)]
            
            metrics = self._measure_performance(
                analyzer.analyze_feature_importance,
                weights, feature_names
            )
            
            # 極大データでも合理的な時間で処理完了することを確認
            self.assertLess(metrics.execution_time, 120,  # 2分以内
                          f"極大データセット処理が時間がかかりすぎます: {metrics.execution_time:.1f}s")
            
            logger.info(f"大規模データセットテスト完了: {n_samples}×{n_features}, "
                       f"時間={metrics.execution_time:.2f}s, メモリ={data_memory:.1f}MB")
    
    def test_memory_leak_detection(self):
        """メモリリーク検出テスト"""
        analyzer = FeatureCorrelationAnalyzer()
        
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        memory_readings = []
        
        # 同じ処理を繰り返し実行
        for i in range(10):
            features, _ = self._generate_test_data(1000, 50)
            
            analyzer.analyze_feature_correlations(features)
            
            # メモリ使用量記録
            current_memory = self.process.memory_info().rss / 1024 / 1024
            memory_readings.append(current_memory)
            
            # 明示的にガベージコレクション
            del features
            gc.collect()
        
        # メモリリーク検出
        memory_trend = memory_readings[-1] - memory_readings[0]
        
        self.assertLess(memory_trend, 50,  # 50MB以上の増加は異常
                       f"メモリリークの可能性があります: {memory_trend:.1f}MB増加")
        
        logger.info(f"メモリリーク検出テスト完了: {memory_trend:.1f}MB変化")


def generate_performance_report(test_results: Dict[str, List[PerformanceMetrics]], 
                              output_dir: Path) -> str:
    """パフォーマンステスト結果レポートを生成"""
    report_file = output_dir / 'performance_report.html'
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>パフォーマンステスト結果レポート</title>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .good {{ background-color: #d4edda; }}
            .warning {{ background-color: #fff3cd; }}
            .bad {{ background-color: #f8d7da; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>パフォーマンステスト結果レポート</h1>
            <p>生成日時: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <div class="section">
            <h2>パフォーマンスサマリー</h2>
            <table>
                <tr><th>テストカテゴリ</th><th>実行回数</th><th>平均実行時間(s)</th><th>平均メモリ使用量(MB)</th><th>平均スループット(/s)</th></tr>
    """
    
    for test_name, metrics_list in test_results.items():
        if metrics_list:
            avg_time = np.mean([m.execution_time for m in metrics_list])
            avg_memory = np.mean([m.memory_usage_mb for m in metrics_list])
            avg_throughput = np.mean([m.throughput for m in metrics_list])
            
            # パフォーマンス評価
            time_class = 'good' if avg_time < 1.0 else 'warning' if avg_time < 10.0 else 'bad'
            memory_class = 'good' if avg_memory < 50 else 'warning' if avg_memory < 200 else 'bad'
            throughput_class = 'good' if avg_throughput > 1000 else 'warning' if avg_throughput > 100 else 'bad'
            
            html_content += f"""
                <tr>
                    <td>{test_name}</td>
                    <td>{len(metrics_list)}</td>
                    <td class="{time_class}">{avg_time:.3f}</td>
                    <td class="{memory_class}">{avg_memory:.1f}</td>
                    <td class="{throughput_class}">{avg_throughput:.1f}</td>
                </tr>
            """
    
    html_content += """
            </table>
        </div>
        
        <div class="section">
            <h2>性能基準</h2>
            <ul>
                <li><span class="good">良好</span>: 実行時間 < 1s, メモリ使用量 < 50MB, スループット > 1000/s</li>
                <li><span class="warning">注意</span>: 実行時間 < 10s, メモリ使用量 < 200MB, スループット > 100/s</li>
                <li><span class="bad">問題</span>: 上記基準を下回る</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return str(report_file)


def run_performance_tests():
    """パフォーマンステストの実行"""
    # テストスイートを作成
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 各テストクラスをスイートに追加
    test_classes = [
        TestFeatureAnalysisPerformance,
        TestFeatureDesignPerformance,
        TestFeatureOptimizationPerformance,
        TestPipelinePerformance,
        TestLargeScaleDatasetPerformance
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
    success_rate = (total_tests - failures - errors) / total_tests * 100 if total_tests > 0 else 0
    
    print(f"\n" + "="*60)
    print("パフォーマンステストスイート実行結果")
    print("="*60)
    print(f"総テスト数: {total_tests}")
    print(f"成功: {total_tests - failures - errors}")
    print(f"失敗: {failures}")
    print(f"エラー: {errors}")
    print(f"成功率: {success_rate:.1f}%")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_performance_tests()
    sys.exit(0 if success else 1)
