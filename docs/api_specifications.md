# API 仕様書

## 目次

1. [概要](#概要)
2. [共通仕様](#共通仕様)
3. [特徴量分析 API](#特徴量分析api)
4. [特徴量設計 API](#特徴量設計api)
5. [特徴量最適化 API](#特徴量最適化api)
6. [GAT 最適化 API](#gat最適化api)
7. [パイプライン API](#パイプラインapi)
8. [エラーハンドリング](#エラーハンドリング)
9. [使用例](#使用例)

## 概要

この API は、IRL（Inverse Reinforcement Learning）特徴量リデザインシステムのプログラマティックインターフェースを提供します。開発者の課題割り当て最適化のための特徴量エンジニアリングを効率的に実行できます。

### システムアーキテクチャ

```
API Layer
├── Analysis APIs (分析)
├── Design APIs (設計)
├── Optimization APIs (最適化)
├── GAT APIs (グラフ注意機構)
└── Pipeline APIs (パイプライン)
```

## 共通仕様

### データ型定義

```python
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

# 基本データ型
FeatureArray = np.ndarray  # Shape: (n_samples, n_features)
LabelArray = np.ndarray    # Shape: (n_samples,)
WeightArray = np.ndarray   # Shape: (n_features,)

# 設定データ型
@dataclass
class APIConfig:
    """API共通設定"""
    api_version: str = "1.0.0"
    timeout_seconds: int = 300
    max_memory_mb: int = 1000
    enable_cache: bool = True
    cache_expiry_hours: int = 24
    log_level: str = "INFO"

# 結果データ型
@dataclass
class APIResult:
    """API共通結果"""
    success: bool
    message: str
    execution_time: float
    memory_usage_mb: float
    data: Dict[str, Any]
    metadata: Dict[str, Any]
```

### 共通パラメータ

| パラメータ        | 型          | 必須 | デフォルト    | 説明           |
| ----------------- | ----------- | ---- | ------------- | -------------- |
| `config`          | `APIConfig` | No   | `APIConfig()` | API 設定       |
| `verbose`         | `bool`      | No   | `False`       | 詳細ログ出力   |
| `validate_input`  | `bool`      | No   | `True`        | 入力データ検証 |
| `enable_parallel` | `bool`      | No   | `True`        | 並列処理有効化 |

### エラーレスポンス

```python
@dataclass
class APIError:
    """API エラー情報"""
    error_code: str
    error_message: str
    error_details: Dict[str, Any]
    stack_trace: Optional[str] = None
    suggestions: List[str] = None
```

## 特徴量分析 API

### FeatureImportanceAnalyzer

#### `analyze_feature_importance`

特徴量重要度を分析します。

**シグネチャ**:

```python
def analyze_feature_importance(
    weights: WeightArray,
    feature_names: List[str],
    analysis_types: List[str] = ['absolute', 'relative', 'category_wise'],
    statistical_test: bool = True,
    significance_level: float = 0.05,
    config: Optional[APIConfig] = None
) -> APIResult
```

**パラメータ**:

- `weights`: IRL 学習済み重み配列
- `feature_names`: 特徴量名のリスト
- `analysis_types`: 分析タイプのリスト
  - `'absolute'`: 絶対重要度
  - `'relative'`: 相対重要度
  - `'category_wise'`: カテゴリ別重要度
  - `'temporal'`: 時系列重要度変化
- `statistical_test`: 統計的有意性検証の有効化
- `significance_level`: 有意水準

**戻り値**:

```python
{
    "success": True,
    "data": {
        "importance_ranking": [
            ("feature_name", importance_score),
            ...
        ],
        "importance_scores": {
            "feature_name1": 0.123,
            "feature_name2": 0.098,
            ...
        },
        "category_importance": {
            "task_features": 0.45,
            "developer_features": 0.35,
            "matching_features": 0.20
        },
        "statistical_significance": {
            "feature_name1": {"p_value": 0.001, "is_significant": True},
            ...
        }
    },
    "metadata": {
        "total_features": 150,
        "significant_features": 42,
        "analysis_methods": ["absolute", "relative", "category_wise"]
    }
}
```

**使用例**:

```python
from analysis.feature_analysis import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer()
weights = np.load('learned_weights.npy')
feature_names = ['task_complexity', 'developer_skill', 'collaboration_history']

result = analyzer.analyze_feature_importance(
    weights=weights,
    feature_names=feature_names,
    analysis_types=['absolute', 'category_wise'],
    statistical_test=True
)

if result.success:
    rankings = result.data['importance_ranking']
    print(f"Top 10 features: {rankings[:10]}")
```

#### `compare_importance_across_models`

複数のモデル間で特徴量重要度を比較します。

**シグネチャ**:

```python
def compare_importance_across_models(
    weights_dict: Dict[str, WeightArray],
    feature_names: List[str],
    comparison_metrics: List[str] = ['correlation', 'rank_correlation', 'consistency'],
    config: Optional[APIConfig] = None
) -> APIResult
```

### FeatureCorrelationAnalyzer

#### `analyze_feature_correlations`

特徴量間の相関関係を分析します。

**シグネチャ**:

```python
def analyze_feature_correlations(
    features: FeatureArray,
    feature_names: List[str],
    correlation_methods: List[str] = ['pearson', 'spearman', 'kendall'],
    high_correlation_threshold: float = 0.8,
    cluster_correlated_features: bool = True,
    config: Optional[APIConfig] = None
) -> APIResult
```

**パラメータ**:

- `features`: 特徴量配列
- `feature_names`: 特徴量名のリスト
- `correlation_methods`: 相関係数計算方法
- `high_correlation_threshold`: 高相関の閾値
- `cluster_correlated_features`: 相関特徴量のクラスタリング有効化

**戻り値**:

```python
{
    "success": True,
    "data": {
        "correlation_matrix": np.ndarray,  # (n_features, n_features)
        "high_correlation_pairs": [
            ("feature1", "feature2", correlation_score),
            ...
        ],
        "redundant_feature_candidates": ["feature3", "feature7", ...],
        "correlation_clusters": {
            "cluster_1": ["feature1", "feature2", "feature3"],
            "cluster_2": ["feature4", "feature5"],
            ...
        },
        "correlation_statistics": {
            "mean_correlation": 0.15,
            "max_correlation": 0.92,
            "highly_correlated_pairs_count": 12
        }
    }
}
```

### FeatureDistributionAnalyzer

#### `analyze_feature_distributions`

特徴量分布を分析します。

**シグネチャ**:

```python
def analyze_feature_distributions(
    features: FeatureArray,
    feature_names: List[str],
    distribution_tests: List[str] = ['normality', 'skewness', 'kurtosis'],
    outlier_detection_methods: List[str] = ['iqr', 'zscore', 'isolation_forest'],
    generate_plots: bool = False,
    config: Optional[APIConfig] = None
) -> APIResult
```

## 特徴量設計 API

### TaskFeatureDesigner

#### `design_enhanced_task_features`

強化されたタスク特徴量を設計します。

**シグネチャ**:

```python
def design_enhanced_task_features(
    task_data: Dict[str, Any],
    feature_types: List[str] = ['urgency', 'complexity', 'social_attention'],
    enhancement_methods: List[str] = ['nlp_analysis', 'time_series', 'graph_analysis'],
    config: Optional[APIConfig] = None
) -> APIResult
```

**パラメータ**:

- `task_data`: タスクデータ辞書
  ```python
  {
      'task_id': List[int],
      'title': List[str],
      'description': List[str],
      'priority': List[str],
      'estimated_hours': List[float],
      'labels': List[List[str]],
      'created_at': List[datetime],
      'deadline': List[datetime]
  }
  ```
- `feature_types`: 生成する特徴量タイプ
- `enhancement_methods`: 強化手法

**戻り値**:

```python
{
    "success": True,
    "data": {
        "urgency_features": {
            "priority_score": np.ndarray,
            "deadline_pressure": np.ndarray,
            "milestone_proximity": np.ndarray,
            "blocking_issue_count": np.ndarray
        },
        "complexity_features": {
            "technical_term_density": np.ndarray,
            "reference_link_count": np.ndarray,
            "estimated_effort_hours": np.ndarray,
            "dependency_count": np.ndarray
        },
        "social_attention_features": {
            "watcher_count": np.ndarray,
            "reaction_count": np.ndarray,
            "mention_count": np.ndarray,
            "external_reference_count": np.ndarray
        },
        "enhanced_features": np.ndarray,  # (n_tasks, n_enhanced_features)
        "feature_names": List[str]
    },
    "metadata": {
        "total_features_generated": 64,
        "feature_categories": {
            "urgency": 12,
            "complexity": 18,
            "social_attention": 8,
            "nlp_derived": 16,
            "temporal": 10
        }
    }
}
```

#### `calculate_task_similarity`

タスク間の類似度を計算します。

**シグネチャ**:

```python
def calculate_task_similarity(
    task_features: FeatureArray,
    similarity_metrics: List[str] = ['cosine', 'euclidean', 'jaccard'],
    clustering_method: str = 'hierarchical',
    config: Optional[APIConfig] = None
) -> APIResult
```

### DeveloperFeatureDesigner

#### `design_enhanced_developer_features`

強化された開発者特徴量を設計します。

**シグネチャ**:

```python
def design_enhanced_developer_features(
    developer_data: Dict[str, Any],
    feature_types: List[str] = ['expertise', 'activity_pattern', 'quality'],
    temporal_window_days: int = 90,
    expertise_domains: List[str] = ['languages', 'frameworks', 'domains'],
    config: Optional[APIConfig] = None
) -> APIResult
```

**パラメータ**:

- `developer_data`: 開発者データ辞書
  ```python
  {
      'developer_id': List[int],
      'commits_count': List[int],
      'expertise_languages': List[List[str]],
      'recent_activity_days': List[int],
      'pr_merge_rate': List[float],
      'code_review_count': List[int],
      'bug_introduction_rate': List[float],
      'collaboration_network': List[List[int]]
  }
  ```

### MatchingFeatureDesigner

#### `design_enhanced_matching_features`

マッチング特徴量を設計します。

**シグネチャ**:

```python
def design_enhanced_matching_features(
    task_data: Dict[str, Any],
    developer_data: Dict[str, Any],
    matching_types: List[str] = ['temporal_proximity', 'technical_compatibility', 'success_history'],
    historical_data_months: int = 12,
    config: Optional[APIConfig] = None
) -> APIResult
```

## 特徴量最適化 API

### FeatureScaler

#### `fit_transform`

特徴量をスケーリングします。

**シグネチャ**:

```python
def fit_transform(
    features: FeatureArray,
    method: str = 'standard',
    feature_types: Optional[List[str]] = None,
    handle_outliers: bool = True,
    config: Optional[APIConfig] = None
) -> APIResult
```

**パラメータ**:

- `method`: スケーリング手法
  - `'standard'`: 標準化 (mean=0, std=1)
  - `'minmax'`: 最小-最大正規化 [0, 1]
  - `'robust'`: ロバストスケーリング (中央値と IQR 使用)
  - `'quantile'`: 分位数変換
  - `'power'`: 冪変換 (Box-Cox, Yeo-Johnson)

#### `auto_scale`

特徴量の性質に応じて自動的にスケーリング手法を選択します。

**シグネチャ**:

```python
def auto_scale(
    features: FeatureArray,
    feature_types: List[str],
    optimization_metric: str = 'ml_performance',
    config: Optional[APIConfig] = None
) -> APIResult
```

### FeatureSelector

#### `select_features`

特徴量選択を実行します。

**シグネチャ**:

```python
def select_features(
    features: FeatureArray,
    labels: LabelArray,
    method: str = 'rfe',
    k: Optional[int] = None,
    selection_criteria: Dict[str, Any] = None,
    config: Optional[APIConfig] = None
) -> APIResult
```

**パラメータ**:

- `method`: 選択手法
  - `'univariate'`: 単変量統計検定
  - `'rfe'`: 再帰的特徴量除去
  - `'lasso'`: L1 正則化
  - `'mutual_info'`: 相互情報量
  - `'stability'`: 安定性選択
- `k`: 選択する特徴量数
- `selection_criteria`: 選択基準の詳細設定

**戻り値**:

```python
{
    "success": True,
    "data": {
        "selected_features": np.ndarray,  # (n_samples, k)
        "selected_indices": np.ndarray,   # (k,)
        "feature_scores": np.ndarray,     # (n_features,)
        "selection_ranking": List[Tuple[int, float]],  # [(index, score), ...]
        "eliminated_features": List[int]
    },
    "metadata": {
        "selection_method": "rfe",
        "original_feature_count": 150,
        "selected_feature_count": 50,
        "selection_time_seconds": 12.5
    }
}
```

### DimensionReducer

#### `reduce_dimensions`

次元削減を実行します。

**シグネチャ**:

```python
def reduce_dimensions(
    features: FeatureArray,
    method: str = 'pca',
    n_components: Optional[int] = None,
    preserve_variance_ratio: float = 0.95,
    config: Optional[APIConfig] = None
) -> APIResult
```

**パラメータ**:

- `method`: 次元削減手法
  - `'pca'`: 主成分分析
  - `'ica'`: 独立成分分析
  - `'tsne'`: t-SNE
  - `'umap'`: UMAP
  - `'autoencoder'`: オートエンコーダー
- `n_components`: 削減後の次元数
- `preserve_variance_ratio`: 保持する分散比率

## GAT 最適化 API

### GATOptimizer

#### `find_optimal_dimensions`

GAT 埋め込みの最適次元数を探索します。

**シグネチャ**:

```python
def find_optimal_dimensions(
    gat_model: Any,  # GAT model instance
    dimension_candidates: List[int] = [16, 32, 64, 128, 256],
    evaluation_metrics: List[str] = ['embedding_quality', 'downstream_performance'],
    cross_validation_folds: int = 5,
    config: Optional[APIConfig] = None
) -> APIResult
```

**戻り値**:

```python
{
    "success": True,
    "data": {
        "optimal_dimensions": 64,
        "quality_scores": {
            16: 0.72,
            32: 0.85,
            64: 0.91,
            128: 0.89,
            256: 0.87
        },
        "detailed_evaluation": {
            "embedding_quality": {...},
            "downstream_performance": {...},
            "computational_cost": {...}
        },
        "recommendation": {
            "primary_choice": 64,
            "alternative_choices": [32, 128],
            "reasoning": "Best trade-off between quality and computational cost"
        }
    }
}
```

#### `optimize_attention_mechanism`

注意機構を最適化します。

**シグネチャ**:

```python
def optimize_attention_mechanism(
    gat_model: Any,
    optimization_targets: List[str] = ['attention_diversity', 'interpretability'],
    regularization_methods: List[str] = ['attention_entropy', 'sparsity'],
    config: Optional[APIConfig] = None
) -> APIResult
```

### GATInterpreter

#### `interpret_gat_dimensions`

GAT 埋め込み次元の意味を解釈します。

**シグネチャ**:

```python
def interpret_gat_dimensions(
    gat_embeddings: FeatureArray,
    feature_names: List[str],
    interpretation_methods: List[str] = ['correlation_analysis', 'clustering', 'projection'],
    config: Optional[APIConfig] = None
) -> APIResult
```

#### `analyze_attention_patterns`

注意パターンを分析します。

**シグネチャ**:

```python
def analyze_attention_patterns(
    attention_weights: FeatureArray,
    node_features: Dict[str, Any],
    pattern_types: List[str] = ['hub_detection', 'community_structure', 'temporal_patterns'],
    config: Optional[APIConfig] = None
) -> APIResult
```

## パイプライン API

### FeaturePipeline

#### `run_full_pipeline`

特徴量エンジニアリングパイプライン全体を実行します。

**シグネチャ**:

```python
def run_full_pipeline(
    input_data: Dict[str, Any],
    pipeline_config: Optional[Dict[str, Any]] = None,
    stages: List[str] = ['analysis', 'design', 'optimization', 'gat_enhancement', 'evaluation'],
    config: Optional[APIConfig] = None
) -> APIResult
```

**パラメータ**:

- `input_data`: 入力データ
  ```python
  {
      'features': FeatureArray,
      'labels': LabelArray,
      'weights': WeightArray,
      'task_data': Dict[str, Any],
      'developer_data': Dict[str, Any],
      'graph_data': Dict[str, Any]
  }
  ```
- `pipeline_config`: パイプライン設定
- `stages`: 実行するステージのリスト

**戻り値**:

```python
{
    "success": True,
    "data": {
        "pipeline_id": "pipeline_20241201_123456",
        "stages_executed": ["analysis", "design", "optimization", "gat_enhancement", "evaluation"],
        "stage_results": {
            "analysis": {...},
            "design": {...},
            "optimization": {...},
            "gat_enhancement": {...},
            "evaluation": {...}
        },
        "final_features": np.ndarray,
        "feature_names": List[str],
        "performance_summary": {
            "original_feature_count": 150,
            "final_feature_count": 85,
            "quality_improvement": 0.23,
            "computational_efficiency_gain": 0.45
        }
    },
    "metadata": {
        "execution_time_seconds": 245.7,
        "memory_usage_mb": 456.2,
        "cache_hit_rate": 0.67,
        "pipeline_version": "1.0.0"
    }
}
```

#### `run_stage`

個別のパイプラインステージを実行します。

**シグネチャ**:

```python
def run_stage(
    stage_name: str,
    stage_input: Dict[str, Any],
    stage_config: Optional[Dict[str, Any]] = None,
    config: Optional[APIConfig] = None
) -> APIResult
```

### FeatureQualityMonitor

#### `monitor_feature_quality`

特徴量品質を継続的に監視します。

**シグネチャ**:

```python
def monitor_feature_quality(
    features: FeatureArray,
    feature_names: List[str],
    quality_metrics: List[str] = ['completeness', 'consistency', 'accuracy', 'timeliness'],
    alert_thresholds: Dict[str, float] = None,
    config: Optional[APIConfig] = None
) -> APIResult
```

### FeatureABTester

#### `run_ab_test`

特徴量の A/B テストを実行します。

**シグネチャ**:

```python
def run_ab_test(
    control_features: FeatureArray,
    treatment_features: FeatureArray,
    labels: LabelArray,
    test_config: Dict[str, Any],
    evaluation_metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
    statistical_tests: List[str] = ['t_test', 'mann_whitney', 'chi_square'],
    config: Optional[APIConfig] = None
) -> APIResult
```

## エラーハンドリング

### エラーコード体系

| コード        | カテゴリ       | 説明                     |
| ------------- | -------------- | ------------------------ |
| `INPUT_001`   | 入力エラー     | 無効な特徴量配列形状     |
| `INPUT_002`   | 入力エラー     | 特徴量名とデータの不整合 |
| `INPUT_003`   | 入力エラー     | 不正なパラメータ値       |
| `COMPUTE_001` | 計算エラー     | メモリ不足               |
| `COMPUTE_002` | 計算エラー     | 数値計算の収束失敗       |
| `COMPUTE_003` | 計算エラー     | 並列処理エラー           |
| `MODEL_001`   | モデルエラー   | GAT 訓練失敗             |
| `MODEL_002`   | モデルエラー   | 特徴量選択失敗           |
| `SYSTEM_001`  | システムエラー | キャッシュアクセス失敗   |
| `SYSTEM_002`  | システムエラー | ファイル I/O エラー      |

### 例外クラス

```python
class FeatureEngineeringError(Exception):
    """特徴量エンジニアリング基底例外"""
    def __init__(self, message: str, error_code: str, details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class InputValidationError(FeatureEngineeringError):
    """入力検証エラー"""
    pass

class ComputationError(FeatureEngineeringError):
    """計算エラー"""
    pass

class ModelError(FeatureEngineeringError):
    """モデル関連エラー"""
    pass
```

### エラーハンドリングの例

```python
try:
    result = analyzer.analyze_feature_importance(weights, feature_names)
except InputValidationError as e:
    print(f"入力エラー [{e.error_code}]: {e.message}")
    if e.details:
        print(f"詳細: {e.details}")
except ComputationError as e:
    print(f"計算エラー [{e.error_code}]: {e.message}")
    # リトライロジック
    time.sleep(1)
    result = analyzer.analyze_feature_importance(weights, feature_names, enable_fallback=True)
except Exception as e:
    print(f"予期しないエラー: {e}")
    # エラー報告
    error_reporter.report_error(e)
```

## 使用例

### 基本的な特徴量分析

```python
from analysis.feature_analysis import FeatureImportanceAnalyzer, FeatureCorrelationAnalyzer
from analysis.feature_design import TaskFeatureDesigner, DeveloperFeatureDesigner
from analysis.feature_optimization import FeatureScaler, FeatureSelector

# 1. 特徴量重要度分析
importance_analyzer = FeatureImportanceAnalyzer()
weights = np.load('learned_weights.npy')
feature_names = load_feature_names()

importance_result = importance_analyzer.analyze_feature_importance(
    weights=weights,
    feature_names=feature_names,
    analysis_types=['absolute', 'category_wise', 'temporal'],
    statistical_test=True
)

print("重要度ランキング:")
for rank, (feature, score) in enumerate(importance_result.data['importance_ranking'][:10]):
    print(f"{rank+1}. {feature}: {score:.4f}")

# 2. 相関分析
correlation_analyzer = FeatureCorrelationAnalyzer()
features = load_features()

correlation_result = correlation_analyzer.analyze_feature_correlations(
    features=features,
    feature_names=feature_names,
    correlation_methods=['pearson', 'spearman'],
    high_correlation_threshold=0.8
)

print(f"高相関ペア数: {len(correlation_result.data['high_correlation_pairs'])}")
print(f"冗長特徴量候補: {correlation_result.data['redundant_feature_candidates']}")

# 3. 新特徴量設計
task_designer = TaskFeatureDesigner()
developer_designer = DeveloperFeatureDesigner()

# タスク特徴量
task_data = load_task_data()
task_features_result = task_designer.design_enhanced_task_features(
    task_data=task_data,
    feature_types=['urgency', 'complexity', 'social_attention'],
    enhancement_methods=['nlp_analysis', 'time_series']
)

# 開発者特徴量
developer_data = load_developer_data()
developer_features_result = developer_designer.design_enhanced_developer_features(
    developer_data=developer_data,
    feature_types=['expertise', 'activity_pattern', 'quality'],
    temporal_window_days=90
)

# 4. 特徴量最適化
scaler = FeatureScaler()
selector = FeatureSelector()

# 結合特徴量
combined_features = np.hstack([
    task_features_result.data['enhanced_features'],
    developer_features_result.data['enhanced_features']
])

# スケーリング
scaled_result = scaler.fit_transform(
    features=combined_features,
    method='auto',
    handle_outliers=True
)

# 特徴量選択
labels = load_labels()
selected_result = selector.select_features(
    features=scaled_result.data['scaled_features'],
    labels=labels,
    method='rfe',
    k=50
)

print(f"最終特徴量数: {selected_result.data['selected_features'].shape[1]}")
```

### パイプライン統合実行

```python
from analysis.feature_pipeline import FeaturePipeline

# パイプライン設定
pipeline_config = {
    'stages': ['analysis', 'design', 'optimization', 'gat_enhancement', 'evaluation'],
    'analysis': {
        'importance_analysis': True,
        'correlation_analysis': True,
        'distribution_analysis': True
    },
    'design': {
        'task_features': True,
        'developer_features': True,
        'matching_features': True
    },
    'optimization': {
        'scaling': True,
        'selection': True,
        'dimension_reduction': True
    },
    'gat_enhancement': {
        'optimization': True,
        'interpretation': True
    }
}

# パイプライン実行
pipeline = FeaturePipeline()
pipeline.initialize_components()

input_data = {
    'features': load_features(),
    'labels': load_labels(),
    'weights': load_weights(),
    'task_data': load_task_data(),
    'developer_data': load_developer_data()
}

result = pipeline.run_full_pipeline(
    input_data=input_data,
    pipeline_config=pipeline_config
)

if result.success:
    print(f"パイプライン実行完了: {result.data['pipeline_id']}")
    print(f"実行時間: {result.metadata['execution_time_seconds']:.2f}秒")
    print(f"最終特徴量形状: {result.data['final_features'].shape}")

    # 結果の保存
    save_pipeline_result(result)
else:
    print(f"パイプライン実行失敗: {result.message}")
```

### A/B テスト実行

```python
from analysis.feature_pipeline import FeatureABTester

ab_tester = FeatureABTester()

# コントロール群（既存特徴量）とトリートメント群（新特徴量）を準備
control_features = load_existing_features()
treatment_features = result.data['final_features']
labels = load_labels()

# A/Bテスト実行
ab_test_config = {
    'test_name': 'enhanced_features_vs_baseline',
    'test_type': 'feature_comparison',
    'min_sample_size': 1000,
    'power': 0.8,
    'alpha': 0.05
}

ab_result = ab_tester.run_ab_test(
    control_features=control_features,
    treatment_features=treatment_features,
    labels=labels,
    test_config=ab_test_config,
    evaluation_metrics=['accuracy', 'precision', 'recall', 'f1'],
    statistical_tests=['t_test', 'mann_whitney']
)

if ab_result.success:
    print("A/Bテスト結果:")
    print(f"統計的有意性: {ab_result.data['statistical_significance']}")
    print(f"性能改善: {ab_result.data['performance_improvement']}")
    print(f"推奨アクション: {ab_result.data['recommendation']}")
```

この API 仕様書は、IRL 特徴量リデザインシステムの全機能へのプログラマティックアクセスを提供します。各 API は一貫したインターフェースと充実したエラーハンドリングを提供し、実用的な特徴量エンジニアリングワークフローを支援します。
