# 特徴量設計ガイド & ベストプラクティス

## 目次

1. [概要](#概要)
2. [特徴量設計の基本原則](#特徴量設計の基本原則)
3. [モジュール別ガイド](#モジュール別ガイド)
4. [ベストプラクティス](#ベストプラクティス)
5. [トラブルシューティング](#トラブルシューティング)
6. [パフォーマンス最適化](#パフォーマンス最適化)

## 概要

このガイドは、IRL（Inverse Reinforcement Learning）特徴量リデザインシステムを使用した特徴量設計の包括的な解説書です。開発者の課題割り当て最適化のための特徴量エンジニアリングを効率的に行うための手法、ベストプラクティス、および具体的な実装例を提供します。

### システム構成

```
特徴量システム階層構造
├── 分析基盤 (Analysis Foundation)
│   ├── FeatureImportanceAnalyzer
│   ├── FeatureCorrelationAnalyzer
│   └── FeatureDistributionAnalyzer
├── 特徴量設計 (Feature Design)
│   ├── TaskFeatureDesigner
│   ├── DeveloperFeatureDesigner
│   └── MatchingFeatureDesigner
├── 最適化 (Optimization)
│   ├── FeatureScaler
│   ├── FeatureSelector
│   └── DimensionReducer
├── GAT最適化 (GAT Optimization)
│   ├── GATOptimizer
│   ├── GATInterpreter
│   └── GATIntegratedOptimizer
└── パイプライン (Pipeline)
    ├── FeaturePipeline
    ├── FeatureQualityMonitor
    └── FeatureABTester
```

## 特徴量設計の基本原則

### 1. ドメイン知識の活用

**原則**: ソフトウェア開発プロセスの深い理解に基づいて特徴量を設計する

```python
# 良い例: ドメイン知識を活用した特徴量
def calculate_urgency_score(priority, deadline_days, blocking_issues):
    """
    緊急度スコアをドメイン知識に基づいて計算
    - 優先度: high=3, medium=2, low=1
    - 期限: 近いほど高スコア
    - ブロッキング課題: 多いほど高スコア
    """
    priority_weight = {'high': 3, 'medium': 2, 'low': 1}
    urgency = priority_weight.get(priority, 1)
    urgency += max(0, (7 - deadline_days) / 7)  # 1週間以内なら追加スコア
    urgency += min(blocking_issues * 0.5, 2.0)  # ブロッキング課題による追加
    return urgency
```

### 2. データ品質の確保

**原則**: 欠損値、外れ値、スケールの不均衡に適切に対処する

```python
# データ品質チェックの例
def validate_feature_quality(features, feature_names):
    quality_report = {}

    for i, name in enumerate(feature_names):
        feature_data = features[:, i]

        quality_report[name] = {
            'missing_ratio': np.isnan(feature_data).mean(),
            'infinite_ratio': np.isinf(feature_data).mean(),
            'unique_ratio': len(np.unique(feature_data)) / len(feature_data),
            'outlier_ratio': detect_outliers_iqr(feature_data).mean(),
            'scale_range': np.ptp(feature_data)  # peak-to-peak range
        }

    return quality_report
```

### 3. 特徴量の解釈可能性

**原則**: 特徴量の意味と重要性を明確に定義し、説明可能にする

```python
# 解釈可能な特徴量設計の例
class InterpretableFeatureDesigner:
    def __init__(self):
        self.feature_definitions = {
            'developer_expertise_score': {
                'description': '開発者の専門性スコア',
                'calculation': 'primary_language_strength * domain_expertise * diversity_bonus',
                'range': [0, 10],
                'interpretation': '高いほど該当分野での専門性が高い'
            }
        }

    def get_feature_explanation(self, feature_name):
        return self.feature_definitions.get(feature_name, {})
```

### 4. スケーラビリティの考慮

**原則**: データサイズの増加に対してリニア・スケーラブルな設計を心がける

```python
# スケーラブルな特徴量計算の例
def calculate_collaboration_features_vectorized(developer_ids, task_history):
    """
    ベクトル化による高速な協力関係特徴量計算
    O(n²) → O(n log n) の計算量改善
    """
    # pandas groupby を活用した効率的な集約
    collaboration_matrix = (
        task_history.groupby(['developer_id', 'collaborator_id'])
        .agg({
            'success_rate': 'mean',
            'interaction_count': 'sum',
            'avg_completion_time': 'mean'
        })
        .reset_index()
    )

    return collaboration_matrix
```

## モジュール別ガイド

### 1. 特徴量分析基盤 (Feature Analysis Foundation)

#### FeatureImportanceAnalyzer

**目的**: IRL 学習後の重みを分析し、特徴量の重要度を定量的に評価

**使用例**:

```python
from analysis.feature_analysis import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer()

# IRL重みファイルから特徴量重要度を分析
weights = np.load('learned_weights.npy')
feature_names = ['task_complexity', 'developer_skill', 'collaboration_history']

importance_result = analyzer.analyze_feature_importance(weights, feature_names)

print("重要度ランキング:")
for rank, (feature, score) in enumerate(importance_result['importance_ranking'][:10]):
    print(f"{rank+1}. {feature}: {score:.4f}")
```

**ベストプラクティス**:

- 重みの正規化を適切に行う
- カテゴリ別重要度比較で特徴量グループの傾向を把握
- 統計的有意性検証で信頼性の高い重要度のみを採用

#### FeatureCorrelationAnalyzer

**目的**: 特徴量間の相関関係を分析し、冗長性を特定

**使用例**:

```python
from analysis.feature_analysis import FeatureCorrelationAnalyzer

analyzer = FeatureCorrelationAnalyzer()

# 相関分析実行
correlation_result = analyzer.analyze_feature_correlations(features)

# 高相関ペアの確認
high_corr_pairs = correlation_result['high_correlation_pairs']
print(f"高相関ペア数: {len(high_corr_pairs)}")

# 冗長特徴量候補の特定
redundant_candidates = correlation_result['redundant_feature_candidates']
print(f"冗長特徴量候補: {redundant_candidates}")
```

**ベストプラクティス**:

- 相関係数 |r| > 0.8 を冗長性の基準とする
- ドメイン知識を活用して、相関があっても有意味な特徴量は保持
- 段階的に冗長特徴量を除去し、性能への影響を確認

#### FeatureDistributionAnalyzer

**目的**: 各特徴量の分布特性を分析し、前処理の必要性を判定

**使用例**:

```python
from analysis.feature_analysis import FeatureDistributionAnalyzer

analyzer = FeatureDistributionAnalyzer()

# 分布分析実行
distribution_result = analyzer.analyze_feature_distributions(features)

# 正規性テスト結果の確認
normality_tests = distribution_result['normality_tests']
non_normal_features = [name for name, is_normal in normality_tests.items() if not is_normal]

print(f"非正規分布特徴量: {non_normal_features}")

# スケール不均衡の確認
scale_issues = distribution_result['scale_imbalance_features']
print(f"スケール調整が必要な特徴量: {scale_issues}")
```

**ベストプラクティス**:

- 正規性テストで p 値 < 0.05 の特徴量は変換を検討
- 歪度・尖度が極端な特徴量には対数変換を適用
- 外れ値の割合が 5%を超える場合は外れ値処理を実施

### 2. 特徴量設計 (Feature Design)

#### TaskFeatureDesigner

**目的**: タスクに関する包括的な特徴量を設計・生成

**使用例**:

```python
from analysis.feature_design import TaskFeatureDesigner

designer = TaskFeatureDesigner()

# タスクデータの準備
task_data = {
    'task_id': [1, 2, 3],
    'title': ['Fix critical bug', 'Add user authentication', 'Update documentation'],
    'description': ['Urgent bug fix needed...', 'Implement OAuth2...', 'Update API docs...'],
    'priority': ['high', 'medium', 'low'],
    'estimated_hours': [8, 40, 4],
    'labels': [['bug', 'critical'], ['feature', 'security'], ['docs']]
}

# 強化された特徴量を生成
enhanced_features = designer.design_enhanced_task_features(task_data)

# 生成された特徴量の確認
print("緊急度特徴量:", enhanced_features['urgency_features'].keys())
print("複雑度特徴量:", enhanced_features['complexity_features'].keys())
print("社会的注目度特徴量:", enhanced_features['social_attention_features'].keys())
```

**設計指針**:

1. **緊急度特徴量 (Urgency Features)**:

   - `priority_score`: 優先度を数値化（high=3, medium=2, low=1）
   - `deadline_pressure`: 期限までの日数の逆数
   - `milestone_proximity`: マイルストーンまでの近接度
   - `blocking_issue_count`: ブロッキング課題の数

2. **複雑度特徴量 (Complexity Features)**:

   - `technical_term_density`: 技術用語の密度
   - `reference_link_count`: 参照リンクの数
   - `estimated_effort_hours`: 推定工数
   - `dependency_count`: 依存関係の数

3. **社会的注目度特徴量 (Social Attention Features)**:
   - `watcher_count`: ウォッチャー数
   - `reaction_count`: リアクション（いいね、コメント）数
   - `mention_count`: メンション数
   - `external_reference_count`: 外部参照数

#### DeveloperFeatureDesigner

**目的**: 開発者の能力、経験、活動パターンに関する特徴量を設計

**使用例**:

```python
from analysis.feature_design import DeveloperFeatureDesigner

designer = DeveloperFeatureDesigner()

# 開発者データの準備
developer_data = {
    'developer_id': [1, 2, 3],
    'commits_count': [1200, 800, 2000],
    'expertise_languages': [['Python', 'JavaScript'], ['Java', 'Kotlin'], ['Python', 'Go', 'Rust']],
    'recent_activity_days': [30, 15, 45],
    'pr_merge_rate': [0.85, 0.92, 0.78],
    'code_review_count': [150, 200, 300],
    'bug_introduction_rate': [0.02, 0.01, 0.03]
}

# 強化された開発者特徴量を生成
enhanced_features = designer.design_enhanced_developer_features(developer_data)

print("専門性特徴量:", enhanced_features['expertise_features'].keys())
print("活動パターン特徴量:", enhanced_features['activity_pattern_features'].keys())
print("品質特徴量:", enhanced_features['quality_features'].keys())
```

**設計指針**:

1. **専門性特徴量 (Expertise Features)**:

   - `primary_language_strength`: 主要言語での強度
   - `domain_expertise_score`: ドメイン専門性スコア
   - `technical_diversity_index`: 技術多様性指標
   - `learning_velocity`: 学習速度（新技術の習得速度）

2. **活動パターン特徴量 (Activity Pattern Features)**:

   - `timezone_activity_pattern`: タイムゾーン別活動パターン
   - `weekday_activity_ratio`: 平日活動比率
   - `response_time_avg`: 平均応答時間
   - `consistency_score`: 活動の一貫性スコア

3. **品質特徴量 (Quality Features)**:
   - `pr_merge_rate`: プルリクエストマージ率
   - `review_approval_rate`: レビュー承認率
   - `bug_introduction_rate`: バグ導入率
   - `code_review_quality_score`: コードレビュー品質スコア

#### MatchingFeatureDesigner

**目的**: タスクと開発者の適合性を測る特徴量を設計

**使用例**:

```python
from analysis.feature_design import MatchingFeatureDesigner

designer = MatchingFeatureDesigner()

# タスクと開発者データを組み合わせて特徴量生成
matching_features = designer.design_enhanced_matching_features(task_data, developer_data)

print("時間的近接性特徴量:", matching_features['temporal_proximity_features'].keys())
print("技術的適合性特徴量:", matching_features['technical_compatibility_features'].keys())
print("成功履歴特徴量:", matching_features['success_history_features'].keys())
```

**設計指針**:

1. **時间的近接性特徴量 (Temporal Proximity Features)**:

   - `recent_collaboration_days`: 最近の協力からの日数
   - `activity_time_overlap`: 活動時間の重複度
   - `timezone_compatibility`: タイムゾーン適合性
   - `response_time_prediction`: 応答時間予測

2. **技術的適合性特徴量 (Technical Compatibility Features)**:

   - `tech_stack_overlap_ratio`: 技術スタック重複率
   - `language_proficiency_match`: 言語習熟度マッチング
   - `framework_experience_score`: フレームワーク経験スコア
   - `architecture_affinity`: アーキテクチャ親和性

3. **成功履歴特徴量 (Success History Features)**:
   - `past_success_rate`: 過去の成功率
   - `similar_task_completion_rate`: 類似タスク完了率
   - `collaboration_satisfaction_score`: 協力満足度スコア
   - `success_probability_estimate`: 成功確率推定

### 3. 特徴量最適化 (Feature Optimization)

#### FeatureScaler

**目的**: 特徴量の値域を統一し、学習アルゴリズムの性能を向上

**使用例**:

```python
from analysis.feature_optimization import FeatureScaler

scaler = FeatureScaler()

# 様々なスケーリング手法を適用
scaled_features_standard = scaler.fit_transform(features, method='standard')
scaled_features_minmax = scaler.fit_transform(features, method='minmax')
scaled_features_robust = scaler.fit_transform(features, method='robust')

# 自動スケーリング選択
auto_scaled_features = scaler.auto_scale(features, feature_types)
```

**スケーリング手法の選択指針**:

| 特徴量タイプ         | 推奨手法                | 理由                              |
| -------------------- | ----------------------- | --------------------------------- |
| 正規分布に近い連続値 | Standard Scaling        | 平均 0、分散 1 に正規化           |
| 一様分布の連続値     | Min-Max Scaling         | [0,1]区間への線形変換             |
| 外れ値を含む連続値   | Robust Scaling          | 中央値と IQR を使用し外れ値に頑健 |
| カテゴリカル値       | One-Hot Encoding        | カテゴリを二進特徴量に変換        |
| 時系列値             | Relative Time Transform | 基準時点からの相対時間            |

#### FeatureSelector

**目的**: 予測性能に寄与する重要な特徴量を自動選択

**使用例**:

```python
from analysis.feature_optimization import FeatureSelector

selector = FeatureSelector()

# 様々な選択手法を試行
selected_features_univariate, indices = selector.select_features(
    features, labels, method='univariate', k=50
)

selected_features_rfe, indices = selector.select_features(
    features, labels, method='rfe', k=30
)

selected_features_lasso, indices = selector.select_features(
    features, labels, method='lasso', alpha=0.01
)

# 重要度ベース選択
importance_based_features = selector.select_by_importance(
    features, labels, importance_threshold=0.01
)
```

**選択手法の特徴**:

1. **単変量統計検定 (Univariate Statistical Tests)**:

   - **Chi-square**: カテゴリカル特徴量 × カテゴリカルターゲット
   - **ANOVA F-test**: 連続特徴量 × カテゴリカルターゲット
   - **相互情報量**: 任意のタイプの組み合わせ

2. **再帰的特徴量除去 (Recursive Feature Elimination)**:

   - モデルベースの重要度を利用
   - 段階的に最低重要度特徴量を除去
   - 計算コストは高いが精度が良い

3. **L1 正則化 (LASSO)**:
   - 自動的にスパースな解を生成
   - 正則化パラメータ α の調整が重要
   - 線形関係に適している

#### DimensionReducer

**目的**: 特徴量の次元数を削減し、計算効率と汎化性能を向上

**使用例**:

```python
from analysis.feature_optimization import DimensionReducer

reducer = DimensionReducer()

# PCAによる次元削減
pca_features = reducer.reduce_dimensions(features, method='pca', n_components=50)

# UMAPによる非線形次元削減
umap_features = reducer.reduce_dimensions(features, method='umap', n_components=32)

# 最適次元数の自動決定
optimal_dims = reducer.find_optimal_dimensions(features, method='pca')
print(f"最適次元数: {optimal_dims}")

# 次元削減後の解釈
interpretation = reducer.interpret_reduced_dimensions(pca_features, original_feature_names)
```

**次元削減手法の比較**:

| 手法  | 利点                       | 欠点                 | 適用場面           |
| ----- | -------------------------- | -------------------- | ------------------ |
| PCA   | 高速、線形変換、解釈可能   | 線形関係のみ         | 前処理、可視化     |
| t-SNE | 局所構造保持、可視化に優秀 | 遅い、パラメータ敏感 | データ探索、可視化 |
| UMAP  | 高速、グローバル構造保持   | パラメータ調整必要   | 実用的な次元削減   |

### 4. GAT 最適化 (GAT Optimization)

#### GATOptimizer

**目的**: Graph Attention Network の埋め込み次元を最適化

**使用例**:

```python
from analysis.gat_optimization import GATOptimizer

optimizer = GATOptimizer()

# GAT모델 최적 차원 찾기
optimal_dims_result = optimizer.find_optimal_dimensions(
    gat_model, dimension_candidates=[16, 32, 64, 128]
)

print(f"최적 차원 수: {optimal_dims_result['optimal_dimensions']}")
print(f"품질 점수: {optimal_dims_result['quality_scores']}")

# 어텐션 가중치 분석
attention_analysis = optimizer.analyze_attention_weights(gat_model)
print(f"중요한 관계 수: {len(attention_analysis['important_relationships'])}")
```

**最適化指針**:

- 埋め込み品質メトリクス（分散、情報量、クラスタリング品質）を総合評価
- 計算コストと精度のトレードオフを考慮
- アテンション重みの解釈可能性を重視

#### GATInterpreter

**目的**: GAT 特徴量の各次元の意味を解釈

**使用例**:

```python
from analysis.gat_optimization import GATInterpreter

interpreter = GATInterpreter()

# GAT次元の解釈
dimension_interpretation = interpreter.interpret_gat_dimensions(
    gat_embeddings, feature_names
)

# 重要なグラフパターンの特定
graph_patterns = interpreter.identify_important_graph_patterns(
    gat_model, collaboration_graph
)

# 協力関係の可視化
collaboration_viz = interpreter.visualize_collaboration_network(
    gat_embeddings, developer_ids, output_path='collaboration_network.png'
)
```

#### GATIntegratedOptimizer

**目的**: 基本特徴量と GAT 特徴量の最適な組み合わせを探索

**使用例**:

```python
from analysis.gat_optimization import GATIntegratedOptimizer

integrated_optimizer = GATIntegratedOptimizer()

# 최적 조합 탐색
optimal_combination = integrated_optimizer.search_optimal_feature_combinations(
    basic_features, gat_features, labels
)

print(f"최적 조합: {optimal_combination['feature_combination']}")
print(f"성능 점수: {optimal_combination['performance_score']}")

# 중복성 제거
redundancy_analysis = integrated_optimizer.remove_feature_redundancy(
    combined_features, correlation_threshold=0.8
)
```

### 5. パイプライン (Pipeline)

#### FeaturePipeline

**目的**: 特徴量エンジニアリングプロセス全体を自動化

**YAML 設定例**:

```yaml
# feature_pipeline_config.yaml
pipeline:
  stages:
    ["analysis", "design", "optimization", "gat_enhancement", "evaluation"]
  enable_cache: true
  cache_expiry_hours: 24
  error_handling: "continue"

analysis:
  importance_analysis: true
  correlation_analysis: true
  distribution_analysis: true
  importance_threshold: 0.01
  correlation_threshold: 0.8

design:
  task_features: true
  developer_features: true
  matching_features: true
  feature_combinations: ["basic", "enhanced", "full"]

optimization:
  scaling: true
  selection: true
  dimension_reduction: true
  scaling_methods: ["standard", "minmax", "robust"]
  selection_methods: ["univariate", "rfe", "lasso"]
  reduction_methods: ["pca", "umap"]

gat_enhancement:
  optimization: true
  interpretation: true
  integration: true
  dimension_analysis: true

evaluation:
  quality_metrics: true
  performance_comparison: true
  generate_reports: true
```

**使用例**:

```python
from analysis.feature_pipeline import FeaturePipeline

# 파이프라인 초기화
pipeline = FeaturePipeline(
    config_path='feature_pipeline_config.yaml',
    cache_dir='./pipeline_cache'
)

pipeline.initialize_components()

# 전체 파이프라인 실행
input_data = {
    'features': raw_features,
    'labels': labels,
    'weights': irl_weights
}

result = pipeline.run_full_pipeline(input_data)

print(f"파이프라인 ID: {result['pipeline_id']}")
print(f"실행된 단계: {result['stages_executed']}")
print(f"성능 요약: {result['performance_summary']}")
```

## ベストプラクティス

### 1. データ前処理のベストプラクティス

#### 欠損値処理

```python
def handle_missing_values(features, strategy='median'):
    """
    欠損値を適切に処理

    strategies:
    - 'median': 中央値で補完（数値特徴量）
    - 'mode': 最頻値で補完（カテゴリカル特徴量）
    - 'forward_fill': 前方補完（時系列特徴量）
    - 'interpolate': 線形補間（連続的な時系列）
    """
    if strategy == 'median':
        return np.where(np.isnan(features), np.nanmedian(features, axis=0), features)
    elif strategy == 'mode':
        # カテゴリカル特徴量の最頻値補完
        pass
    # その他の戦略...
```

#### 外れ値検出と処理

```python
def detect_and_handle_outliers(features, method='iqr', action='cap'):
    """
    外れ値を検出して処理

    methods:
    - 'iqr': 四分位範囲ベース
    - 'zscore': Zスコアベース
    - 'isolation_forest': 異常検知手法

    actions:
    - 'cap': 上下限値でクリッピング
    - 'remove': 外れ値を含む行を除去
    - 'transform': 対数変換等で緩和
    """
    if method == 'iqr':
        Q1 = np.percentile(features, 25, axis=0)
        Q3 = np.percentile(features, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if action == 'cap':
            return np.clip(features, lower_bound, upper_bound)
    # その他の方法...
```

### 2. 特徴量選択のベストプラクティス

#### 段階的特徴量選択

```python
def staged_feature_selection(features, labels, stages):
    """
    段階的に特徴量を選択してオーバーフィッティングを防止

    stages: [
        {'method': 'variance', 'threshold': 0.01},
        {'method': 'correlation', 'threshold': 0.95},
        {'method': 'univariate', 'k': 100},
        {'method': 'rfe', 'k': 50}
    ]
    """
    current_features = features.copy()
    selection_history = []

    for stage in stages:
        if stage['method'] == 'variance':
            # 低分散特徴量除去
            selector = VarianceThreshold(threshold=stage['threshold'])
            current_features = selector.fit_transform(current_features)

        elif stage['method'] == 'correlation':
            # 高相関特徴量除去
            corr_matrix = np.corrcoef(current_features.T)
            # 相関の高いペアの一方を除去

        # 他のステージ...

        selection_history.append({
            'stage': stage['method'],
            'features_before': current_features.shape[1],
            'features_after': current_features.shape[1]
        })

    return current_features, selection_history
```

#### クロスバリデーションベース選択

```python
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def cv_based_feature_selection(features, labels, cv_folds=5):
    """
    クロスバリデーションベースの特徴量選択
    """
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 異なる特徴量数での性能を評価
    feature_counts = range(10, min(features.shape[1], 101), 10)
    cv_scores = []

    for n_features in feature_counts:
        selector = RFE(base_model, n_features_to_select=n_features)
        selected_features = selector.fit_transform(features, labels)

        scores = cross_val_score(base_model, selected_features, labels, cv=cv_folds)
        cv_scores.append(scores.mean())

    # 最適な特徴量数を特定
    optimal_n_features = feature_counts[np.argmax(cv_scores)]

    return optimal_n_features, cv_scores
```

### 3. GAT 最適化のベストプラクティス

#### 段階的次元最適化

```python
def staged_gat_optimization(gat_model, feature_data, dimension_range):
    """
    段階的にGAT次元を最適化
    """
    optimization_history = []

    # 粗い探索
    coarse_dims = range(dimension_range[0], dimension_range[1], 16)
    coarse_results = []

    for dim in coarse_dims:
        gat_model.set_embedding_dim(dim)
        quality_score = evaluate_embedding_quality(gat_model, feature_data)
        coarse_results.append((dim, quality_score))

    # 最良の範囲で細かい探索
    best_coarse_dim = max(coarse_results, key=lambda x: x[1])[0]
    fine_range = range(max(dimension_range[0], best_coarse_dim - 8),
                      min(dimension_range[1], best_coarse_dim + 9), 2)

    fine_results = []
    for dim in fine_range:
        gat_model.set_embedding_dim(dim)
        quality_score = evaluate_embedding_quality(gat_model, feature_data)
        fine_results.append((dim, quality_score))

    optimal_dim = max(fine_results, key=lambda x: x[1])[0]

    return optimal_dim, optimization_history
```

### 4. パイプライン設計のベストプラクティス

#### エラーハンドリングと復旧

```python
class RobustFeaturePipeline:
    def __init__(self):
        self.fallback_strategies = {
            'analysis': self._analysis_fallback,
            'design': self._design_fallback,
            'optimization': self._optimization_fallback
        }

    def execute_stage_with_fallback(self, stage_name, stage_func, *args, **kwargs):
        """
        フォールバック機能付きステージ実行
        """
        try:
            return stage_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Stage {stage_name} failed: {e}. Using fallback.")
            fallback_func = self.fallback_strategies.get(stage_name)
            if fallback_func:
                return fallback_func(*args, **kwargs)
            else:
                raise

    def _analysis_fallback(self, *args, **kwargs):
        """分析ステージのフォールバック"""
        return {
            'importance_ranking': [],
            'correlation_matrix': np.eye(kwargs.get('n_features', 10)),
            'distribution_stats': {}
        }
```

#### キャッシュ効率化

```python
import hashlib
import pickle
from functools import wraps

def cache_pipeline_stage(cache_dir, expire_hours=24):
    """
    パイプラインステージの結果をキャッシュ
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # キャッシュキーを生成
            cache_key = hashlib.md5(
                f"{func.__name__}_{str(args)}_{str(sorted(kwargs.items()))}".encode()
            ).hexdigest()

            cache_file = Path(cache_dir) / f"{cache_key}.pkl"

            # キャッシュチェック
            if cache_file.exists():
                cache_age = time.time() - cache_file.stat().st_mtime
                if cache_age < expire_hours * 3600:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)

            # 実際の処理実行
            result = func(*args, **kwargs)

            # キャッシュに保存
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)

            return result
        return wrapper
    return decorator
```

## トラブルシューティング

### よくある問題と解決方法

#### 1. メモリ不足エラー

**問題**: 大規模データセットでメモリ不足が発生

**解決方法**:

```python
# バッチ処理による解決
def process_large_dataset_in_batches(features, batch_size=1000):
    """
    大規模データセットをバッチ処理
    """
    n_samples = features.shape[0]
    results = []

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_features = features[start_idx:end_idx]

        # バッチ処理
        batch_result = process_batch(batch_features)
        results.append(batch_result)

        # メモリクリーンアップ
        del batch_features
        gc.collect()

    return np.concatenate(results, axis=0)
```

#### 2. 特徴量スケールの不均衡

**問題**: 異なるスケールの特徴量が混在

**診断**:

```python
def diagnose_scale_imbalance(features, feature_names):
    """
    スケール不均衡を診断
    """
    scales = []
    for i, name in enumerate(feature_names):
        feature_range = np.ptp(features[:, i])  # peak-to-peak
        feature_std = np.std(features[:, i])
        scales.append({
            'feature': name,
            'range': feature_range,
            'std': feature_std,
            'coefficient_of_variation': feature_std / np.mean(features[:, i]) if np.mean(features[:, i]) != 0 else np.inf
        })

    # スケール差が大きい特徴量を特定
    scales.sort(key=lambda x: x['range'], reverse=True)
    return scales
```

**解決方法**:

```python
# 自動スケール選択
def auto_select_scaling_method(feature_data):
    """
    特徴量の特性に応じてスケーリング手法を自動選択
    """
    # 分布の正規性をテスト
    _, p_value = stats.normaltest(feature_data)

    # 外れ値の割合を計算
    Q1, Q3 = np.percentile(feature_data, [25, 75])
    IQR = Q3 - Q1
    outlier_ratio = np.mean((feature_data < Q1 - 1.5*IQR) | (feature_data > Q3 + 1.5*IQR))

    if p_value > 0.05 and outlier_ratio < 0.05:
        return 'standard'  # 正規分布で外れ値が少ない
    elif outlier_ratio > 0.1:
        return 'robust'    # 外れ値が多い
    else:
        return 'minmax'    # その他の場合
```

#### 3. 特徴量選択の不安定性

**問題**: 異なる実行で選択される特徴量が変わる

**解決方法**:

```python
def stable_feature_selection(features, labels, n_trials=10, stability_threshold=0.7):
    """
    複数回実行して安定した特徴量を選択
    """
    feature_selection_counts = np.zeros(features.shape[1])

    for trial in range(n_trials):
        # ブートストラップサンプリング
        indices = np.random.choice(len(features), size=len(features), replace=True)
        bootstrap_features = features[indices]
        bootstrap_labels = labels[indices]

        # 特徴量選択実行
        selector = SelectKBest(f_classif, k=50)
        selector.fit(bootstrap_features, bootstrap_labels)

        # 選択された特徴量をカウント
        selected_indices = selector.get_support(indices=True)
        feature_selection_counts[selected_indices] += 1

    # 安定した特徴量（閾値以上の頻度で選択）を特定
    stable_features = feature_selection_counts >= (n_trials * stability_threshold)

    return stable_features, feature_selection_counts / n_trials
```

#### 4. GAT 収束問題

**問題**: GAT 学習が収束しない

**診断と解決**:

```python
def diagnose_gat_convergence(gat_model, training_history):
    """
    GAT収束問題を診断
    """
    losses = training_history['losses']

    # 収束の診断
    recent_losses = losses[-10:]
    loss_variance = np.var(recent_losses)
    loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]

    diagnosis = {
        'is_converged': loss_variance < 0.001 and abs(loss_trend) < 0.001,
        'is_oscillating': loss_variance > 0.01,
        'is_diverging': loss_trend > 0.01,
        'recommended_actions': []
    }

    if diagnosis['is_oscillating']:
        diagnosis['recommended_actions'].append('learning_rate_reduction')
    if diagnosis['is_diverging']:
        diagnosis['recommended_actions'].append('gradient_clipping')

    return diagnosis

def fix_gat_convergence_issues(gat_model, issues):
    """
    GAT収束問題を修正
    """
    if 'learning_rate_reduction' in issues:
        current_lr = gat_model.optimizer.param_groups[0]['lr']
        new_lr = current_lr * 0.5
        gat_model.optimizer.param_groups[0]['lr'] = new_lr

    if 'gradient_clipping' in issues:
        torch.nn.utils.clip_grad_norm_(gat_model.parameters(), max_norm=1.0)
```

## パフォーマンス最適化

### 1. 計算効率の改善

#### ベクトル化の活用

```python
# 非効率な例（ループ使用）
def calculate_similarity_slow(features1, features2):
    similarities = []
    for i in range(len(features1)):
        for j in range(len(features2)):
            sim = cosine_similarity(features1[i], features2[j])
            similarities.append(sim)
    return similarities

# 効率的な例（ベクトル化）
def calculate_similarity_fast(features1, features2):
    # NumPyのブロードキャスティングを活用
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(features1, features2)
```

#### 並列処理の活用

```python
from multiprocessing import Pool
from functools import partial

def parallel_feature_computation(feature_batch, computation_func):
    """
    特徴量計算を並列化
    """
    n_cores = multiprocessing.cpu_count()
    batch_size = len(feature_batch) // n_cores

    with Pool(processes=n_cores) as pool:
        batches = [feature_batch[i:i+batch_size]
                  for i in range(0, len(feature_batch), batch_size)]

        results = pool.map(computation_func, batches)

    return np.concatenate(results, axis=0)
```

### 2. メモリ効率の改善

#### ジェネレータベース処理

```python
def feature_generator(data_source, batch_size=1000):
    """
    メモリ効率的な特徴量生成
    """
    for start_idx in range(0, len(data_source), batch_size):
        end_idx = min(start_idx + batch_size, len(data_source))
        batch_data = data_source[start_idx:end_idx]

        # バッチ処理
        batch_features = process_batch(batch_data)

        yield batch_features

        # メモリクリーンアップ
        del batch_data, batch_features

# 使用例
for feature_batch in feature_generator(large_dataset):
    # バッチごとに処理
    process_feature_batch(feature_batch)
```

#### 適応的バッチサイズ

```python
def adaptive_batch_processing(data, target_memory_mb=500):
    """
    メモリ使用量に応じてバッチサイズを調整
    """
    # 初期バッチサイズ
    batch_size = 1000

    while True:
        try:
            # バッチ処理を試行
            batch = data[:batch_size]
            memory_usage = get_memory_usage_mb()

            if memory_usage > target_memory_mb:
                # バッチサイズを縮小
                batch_size = max(batch_size // 2, 10)
            elif memory_usage < target_memory_mb * 0.7:
                # バッチサイズを拡大
                batch_size = min(batch_size * 2, len(data))

            yield batch

        except MemoryError:
            batch_size = max(batch_size // 2, 10)
            continue
```

### 3. キャッシュ戦略

#### 智能キャッシュシステム

```python
class IntelligentCache:
    def __init__(self, max_memory_mb=1000):
        self.cache = {}
        self.access_counts = {}
        self.max_memory = max_memory_mb * 1024 * 1024
        self.current_memory = 0

    def get(self, key):
        if key in self.cache:
            self.access_counts[key] += 1
            return self.cache[key]
        return None

    def put(self, key, value):
        value_size = sys.getsizeof(value)

        # メモリ制限チェック
        while self.current_memory + value_size > self.max_memory:
            self._evict_least_used()

        self.cache[key] = value
        self.access_counts[key] = 1
        self.current_memory += value_size

    def _evict_least_used(self):
        if not self.cache:
            return

        # 最も使用頻度の低いアイテムを削除
        least_used_key = min(self.access_counts, key=self.access_counts.get)
        value_size = sys.getsizeof(self.cache[least_used_key])

        del self.cache[least_used_key]
        del self.access_counts[least_used_key]
        self.current_memory -= value_size
```

この特徴量設計ガイドは、IRL 特徴量リデザインシステムを効果的に活用するための包括的な指針を提供します。実際の運用では、プロジェクト固有の要件に応じてこのガイドの内容を調整・拡張してください。
