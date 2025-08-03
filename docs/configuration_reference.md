# 設定ファイルリファレンス

## 目次

1. [概要](#概要)
2. [パイプライン設定](#パイプライン設定)
3. [分析モジュール設定](#分析モジュール設定)
4. [設計モジュール設定](#設計モジュール設定)
5. [最適化モジュール設定](#最適化モジュール設定)
6. [GAT 最適化設定](#gat最適化設定)
7. [品質監視設定](#品質監視設定)
8. [A/B テスト設定](#abテスト設定)
9. [環境設定](#環境設定)
10. [設定例](#設定例)

## 概要

IRL 特徴量リデザインシステムでは、YAML 形式の設定ファイルを使用してすべてのコンポーネントの動作をカスタマイズできます。このリファレンスでは、利用可能なすべての設定オプションとその詳細を説明します。

### 設定ファイル構造

```yaml
# メイン設定ファイル: feature_pipeline_config.yaml
pipeline: # パイプライン全体設定
  ...
analysis: # 特徴量分析設定
  ...
design: # 特徴量設計設定
  ...
optimization: # 特徴量最適化設定
  ...
gat_enhancement: # GAT最適化設定
  ...
quality_monitoring: # 品質監視設定
  ...
ab_testing: # A/Bテスト設定
  ...
environment: # 環境設定
  ...
```

## パイプライン設定

### 基本設定 (`pipeline`)

```yaml
pipeline:
  # 実行するステージのリスト
  stages:
    ["analysis", "design", "optimization", "gat_enhancement", "evaluation"]

  # ステージ実行順序の制約
  stage_dependencies:
    design: ["analysis"]
    optimization: ["design"]
    gat_enhancement: ["optimization"]
    evaluation: ["gat_enhancement"]

  # キャッシュ設定
  enable_cache: true
  cache_dir: "./pipeline_cache"
  cache_expiry_hours: 24
  cache_compression: true

  # エラーハンドリング
  error_handling: "continue" # 'stop', 'continue', 'retry'
  max_retries: 3
  retry_delay_seconds: 5

  # 並列処理設定
  enable_parallel: true
  max_workers: 4
  parallel_backend: "threading" # 'threading', 'multiprocessing'

  # ログ設定
  log_level: "INFO" # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
  log_file: "pipeline.log"
  enable_progress_bar: true

  # メモリ管理
  max_memory_mb: 2048
  memory_monitoring: true
  garbage_collection_frequency: 10 # ステージ数
```

**パラメータ詳細**:

| パラメータ           | 型          | デフォルト                                                                | 説明                                                                      |
| -------------------- | ----------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `stages`             | `List[str]` | `['analysis', 'design', 'optimization', 'gat_enhancement', 'evaluation']` | 実行するパイプラインステージ                                              |
| `error_handling`     | `str`       | `'continue'`                                                              | エラー時の動作: `'stop'`(即座停止), `'continue'`(続行), `'retry'`(再試行) |
| `max_workers`        | `int`       | `4`                                                                       | 並列処理時の最大ワーカー数                                                |
| `cache_expiry_hours` | `int`       | `24`                                                                      | キャッシュの有効期限（時間）                                              |

## 分析モジュール設定

### 重要度分析設定 (`analysis.importance`)

```yaml
analysis:
  importance_analysis:
    enabled: true

    # 分析タイプ
    analysis_types:
      - "absolute" # 絶対重要度
      - "relative" # 相対重要度
      - "category_wise" # カテゴリ別重要度
      - "temporal" # 時系列重要度変化

    # 統計的検定
    statistical_test: true
    significance_level: 0.05
    multiple_testing_correction: "bonferroni" # 'bonferroni', 'fdr_bh', 'none'

    # 重要度閾値
    importance_threshold: 0.01
    top_k_features: 50

    # 可視化設定
    generate_plots: true
    plot_format: "png" # 'png', 'svg', 'pdf'
    plot_dpi: 300

    # カテゴリ分類
    feature_categories:
      task_features: ["task_*", "priority_*", "complexity_*"]
      developer_features: ["dev_*", "skill_*", "activity_*"]
      matching_features: ["match_*", "similarity_*", "compatibility_*"]
```

### 相関分析設定 (`analysis.correlation`)

```yaml
analysis:
  correlation_analysis:
    enabled: true

    # 相関係数の種類
    correlation_methods:
      - "pearson" # ピアソン相関
      - "spearman" # スピアマン相関
      - "kendall" # ケンドール相関

    # 閾値設定
    high_correlation_threshold: 0.8
    moderate_correlation_threshold: 0.5

    # クラスタリング設定
    cluster_correlated_features: true
    clustering_method: "hierarchical" # 'hierarchical', 'kmeans'
    linkage_method: "ward" # 'ward', 'complete', 'average'

    # 冗長性除去
    auto_remove_redundant: false
    redundancy_removal_strategy: "variance_based" # 'variance_based', 'importance_based'

    # 出力設定
    save_correlation_matrix: true
    correlation_heatmap: true
    dendogram_plot: true
```

### 分布分析設定 (`analysis.distribution`)

```yaml
analysis:
  distribution_analysis:
    enabled: true

    # 分布検定
    distribution_tests:
      - "normality" # 正規性検定 (Shapiro-Wilk, Kolmogorov-Smirnov)
      - "skewness" # 歪度検定
      - "kurtosis" # 尖度検定
      - "outliers" # 外れ値検定

    # 外れ値検出手法
    outlier_detection_methods:
      - "iqr" # 四分位範囲
      - "zscore" # Zスコア
      - "isolation_forest" # 孤立森林
      - "local_outlier_factor" # 局所外れ値因子

    # 正規性検定設定
    normality_test_method: "shapiro" # 'shapiro', 'kstest', 'jarque_bera'
    normality_alpha: 0.05

    # 外れ値設定
    outlier_threshold: 3.0 # Z-score threshold
    outlier_action: "flag" # 'flag', 'remove', 'cap'

    # ビン設定（ヒストグラム用）
    histogram_bins: 50
    adaptive_binning: true
```

## 設計モジュール設定

### タスク特徴量設計 (`design.task_features`)

```yaml
design:
  task_features:
    enabled: true

    # 生成する特徴量タイプ
    feature_types:
      - "urgency" # 緊急度特徴量
      - "complexity" # 複雑度特徴量
      - "social_attention" # 社会的注目度特徴量
      - "temporal" # 時間的特徴量
      - "semantic" # セマンティック特徴量

    # 強化手法
    enhancement_methods:
      - "nlp_analysis" # 自然言語処理分析
      - "time_series" # 時系列分析
      - "graph_analysis" # グラフ分析
      - "clustering" # クラスタリング

    # NLP設定
    nlp_config:
      language: "english"
      tokenizer: "spacy" # 'spacy', 'nltk', 'transformers'
      embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
      max_sequence_length: 512

    # テキスト特徴量
    text_features:
      tfidf_max_features: 1000
      ngram_range: [1, 3]
      min_df: 2
      max_df: 0.8

    # 緊急度計算
    urgency_calculation:
      priority_weights:
        critical: 4.0
        high: 3.0
        medium: 2.0
        low: 1.0
        none: 0.5
      deadline_pressure_window_days: 7
      milestone_weight: 1.5

    # 複雑度計算
    complexity_calculation:
      technical_term_dictionary: "technical_terms.json"
      complexity_indicators:
        - "code_changes_count"
        - "files_affected_count"
        - "dependency_count"
        - "review_comment_count"
```

### 開発者特徴量設計 (`design.developer_features`)

```yaml
design:
  developer_features:
    enabled: true

    # 特徴量タイプ
    feature_types:
      - "expertise" # 専門性特徴量
      - "activity_pattern" # 活動パターン特徴量
      - "quality" # 品質特徴量
      - "collaboration" # 協力特徴量
      - "learning" # 学習特徴量

    # 時間窓設定
    temporal_window_days: 90
    activity_history_months: 12

    # 専門性評価
    expertise_domains:
      - "languages" # プログラミング言語
      - "frameworks" # フレームワーク
      - "domains" # ドメイン知識
      - "tools" # 開発ツール

    # 専門性計算
    expertise_calculation:
      min_commits_for_expertise: 50
      expertise_decay_rate: 0.1 # 月次減衰率
      diversity_bonus_factor: 1.2

    # 活動パターン分析
    activity_pattern_analysis:
      timezone_analysis: true
      weekday_pattern: true
      response_time_analysis: true
      burst_activity_detection: true

    # 品質メトリクス
    quality_metrics:
      - "pr_merge_rate"
      - "code_review_approval_rate"
      - "bug_introduction_rate"
      - "test_coverage_contribution"
      - "documentation_quality_score"

    # 協力ネットワーク
    collaboration_network:
      min_collaboration_count: 5
      network_centrality_measures:
        - "degree_centrality"
        - "betweenness_centrality"
        - "closeness_centrality"
        - "eigenvector_centrality"
```

### マッチング特徴量設計 (`design.matching_features`)

```yaml
design:
  matching_features:
    enabled: true

    # マッチングタイプ
    matching_types:
      - "temporal_proximity" # 時間的近接性
      - "technical_compatibility" # 技術的適合性
      - "success_history" # 成功履歴
      - "workload_balance" # 作業量バランス
      - "skill_complementarity" # スキル補完性

    # 履歴データ期間
    historical_data_months: 12
    min_historical_interactions: 10

    # 類似度計算
    similarity_metrics:
      - "cosine"
      - "jaccard"
      - "euclidean"
      - "manhattan"

    # 時間的近接性
    temporal_proximity:
      recent_collaboration_weight: 2.0
      time_decay_rate: 0.1
      timezone_compatibility_bonus: 1.5

    # 技術適合性
    technical_compatibility:
      language_match_weight: 3.0
      framework_match_weight: 2.0
      domain_match_weight: 2.5
      tool_match_weight: 1.5

    # 成功履歴
    success_history:
      success_rate_weight: 2.0
      completion_time_weight: 1.5
      quality_score_weight: 2.5
      satisfaction_score_weight: 1.8
```

## 最適化モジュール設定

### スケーリング設定 (`optimization.scaling`)

```yaml
optimization:
  scaling:
    enabled: true

    # スケーリング手法
    methods:
      - "standard" # 標準化
      - "minmax" # 最小-最大正規化
      - "robust" # ロバストスケーリング
      - "quantile" # 分位数変換
      - "power" # 冪変換

    # 自動選択設定
    auto_selection: true
    selection_criteria:
      - "distribution_normality"
      - "outlier_presence"
      - "feature_type"
      - "downstream_performance"

    # 手法別設定
    standard_scaling:
      with_mean: true
      with_std: true

    minmax_scaling:
      feature_range: [0, 1]
      clip: true

    robust_scaling:
      quantile_range: [25.0, 75.0]
      with_centering: true
      with_scaling: true

    power_transformation:
      method: "yeo-johnson" # 'box-cox', 'yeo-johnson'
      standardize: true

    # 特徴量タイプ別設定
    feature_type_mapping:
      continuous: "standard"
      discrete: "robust"
      categorical: "onehot"
      temporal: "relative_time"
```

### 特徴量選択設定 (`optimization.selection`)

```yaml
optimization:
  selection:
    enabled: true

    # 選択手法
    methods:
      - "univariate" # 単変量統計検定
      - "rfe" # 再帰的特徴量除去
      - "lasso" # L1正則化
      - "mutual_info" # 相互情報量
      - "stability" # 安定性選択
      - "permutation" # 順列重要度

    # 目標特徴量数
    target_feature_count: 50
    selection_ratio: 0.3 # 元の特徴量数に対する比率

    # 手法別設定
    univariate_selection:
      statistical_test: "f_classif" # 'f_classif', 'chi2', 'mutual_info_classif'
      k_best: 100

    rfe_selection:
      estimator: "random_forest" # 'random_forest', 'svm', 'logistic_regression'
      step: 0.1
      cv_folds: 5

    lasso_selection:
      alpha: 0.01
      cv_folds: 5
      max_iter: 1000

    stability_selection:
      n_bootstrap_iterations: 100
      threshold: 0.6
      lambda_range: [0.01, 1.0]

    # 評価設定
    evaluation_metric: "f1_score" # 'accuracy', 'precision', 'recall', 'f1_score'
    cross_validation: true
    cv_folds: 5

    # 安定性評価
    stability_evaluation: true
    n_stability_runs: 10
    stability_threshold: 0.8
```

### 次元削減設定 (`optimization.dimension_reduction`)

```yaml
optimization:
  dimension_reduction:
    enabled: true

    # 次元削減手法
    methods:
      - "pca" # 主成分分析
      - "ica" # 独立成分分析
      - "umap" # UMAP
      - "tsne" # t-SNE
      - "autoencoder" # オートエンコーダー

    # 目標次元数
    target_dimensions: 32
    preserve_variance_ratio: 0.95

    # 手法別設定
    pca:
      whiten: false
      svd_solver: "auto" # 'auto', 'full', 'arpack', 'randomized'

    ica:
      max_iter: 200
      tolerance: 1e-4

    umap:
      n_neighbors: 15
      min_dist: 0.1
      metric: "euclidean"

    tsne:
      perplexity: 30.0
      early_exaggeration: 12.0
      learning_rate: 200.0
      max_iter: 1000

    autoencoder:
      hidden_layers: [128, 64, 32]
      activation: "relu"
      optimizer: "adam"
      learning_rate: 0.001
      batch_size: 32
      epochs: 100

    # 最適次元数探索
    dimension_search:
      search_range: [16, 128]
      search_step: 16
      evaluation_metrics:
        - "reconstruction_error"
        - "downstream_performance"
        - "clustering_quality"
```

## GAT 最適化設定

### GAT 基本設定 (`gat_enhancement.optimization`)

```yaml
gat_enhancement:
  optimization:
    enabled: true

    # 次元探索設定
    dimension_candidates: [16, 32, 64, 128, 256]
    dimension_search_strategy: "grid" # 'grid', 'random', 'bayesian'

    # 評価メトリクス
    evaluation_metrics:
      - "embedding_quality"
      - "downstream_performance"
      - "attention_diversity"
      - "computational_efficiency"

    # クロスバリデーション
    cross_validation: true
    cv_folds: 5

    # 最適化目標
    optimization_objective: "multi_objective" # 'single_objective', 'multi_objective'
    objective_weights:
      embedding_quality: 0.4
      downstream_performance: 0.3
      attention_diversity: 0.2
      computational_efficiency: 0.1

    # GAT アーキテクチャ
    gat_architecture:
      num_layers: 2
      hidden_dim: 64
      num_heads: 8
      dropout: 0.1
      attention_dropout: 0.1

    # 訓練設定
    training:
      optimizer: "adam"
      learning_rate: 0.001
      weight_decay: 5e-4
      max_epochs: 200
      early_stopping: true
      patience: 20

    # 正則化
    regularization:
      attention_entropy_weight: 0.01
      sparsity_weight: 0.001
      diversity_weight: 0.01
```

### GAT 解釈設定 (`gat_enhancement.interpretation`)

```yaml
gat_enhancement:
  interpretation:
    enabled: true

    # 解釈手法
    interpretation_methods:
      - "attention_analysis"
      - "embedding_analysis"
      - "gradient_analysis"
      - "perturbation_analysis"

    # 注意分析
    attention_analysis:
      attention_threshold: 0.1
      pattern_detection: true
      community_detection: true
      hub_detection: true

    # 埋め込み分析
    embedding_analysis:
      clustering_methods: ["kmeans", "hierarchical", "dbscan"]
      dimensionality_reduction: ["pca", "tsne", "umap"]
      similarity_metrics: ["cosine", "euclidean"]

    # 摂動分析
    perturbation_analysis:
      perturbation_types: ["node", "edge", "feature"]
      perturbation_ratios: [0.1, 0.2, 0.3]

    # 可視化設定
    visualization:
      network_layout: "spring" # 'spring', 'circular', 'kamada_kawai'
      node_size_attribute: "degree"
      edge_width_attribute: "attention_weight"
      color_scheme: "viridis"
      figure_size: [12, 8]
      save_format: "png"
```

### GAT 統合設定 (`gat_enhancement.integration`)

```yaml
gat_enhancement:
  integration:
    enabled: true

    # 統合戦略
    integration_strategies:
      - "concatenation" # 単純結合
      - "weighted_sum" # 重み付き和
      - "attention_fusion" # 注意融合
      - "hierarchical" # 階層統合

    # 重み学習
    weight_learning:
      method: "learned" # 'learned', 'fixed', 'adaptive'
      initialization: "uniform" # 'uniform', 'random', 'importance_based'
      regularization: 0.01

    # 冗長性除去
    redundancy_removal:
      correlation_threshold: 0.8
      importance_threshold: 0.01
      removal_strategy: "greedy" # 'greedy', 'optimal'

    # 品質保証
    quality_assurance:
      validation_split: 0.2
      performance_threshold: 0.95 # 元性能に対する比率
      stability_test: true
      stability_runs: 10
```

## 品質監視設定

### 品質メトリクス (`quality_monitoring.metrics`)

```yaml
quality_monitoring:
  metrics:
    enabled: true

    # 監視メトリクス
    quality_metrics:
      - "completeness" # 完全性
      - "consistency" # 一貫性
      - "accuracy" # 正確性
      - "timeliness" # 適時性
      - "validity" # 妥当性
      - "uniqueness" # 一意性
      - "relevance" # 関連性
      - "stability" # 安定性

    # メトリクス詳細設定
    completeness:
      missing_value_threshold: 0.05 # 5%以下

    consistency:
      format_validation: true
      range_validation: true

    accuracy:
      outlier_detection: true
      outlier_threshold: 3.0

    timeliness:
      freshness_threshold_hours: 24

    validity:
      schema_validation: true
      business_rule_validation: true

    uniqueness:
      duplicate_threshold: 0.01

    relevance:
      correlation_threshold: 0.1
      importance_threshold: 0.01

    stability:
      distribution_shift_threshold: 0.1
      statistical_test: "ks_test"

    # アラート設定
    alerts:
      enable_alerts: true
      alert_channels: ["email", "slack"]
      severity_levels: ["low", "medium", "high", "critical"]

      # アラート閾値
      alert_thresholds:
        completeness: 0.95
        consistency: 0.90
        accuracy: 0.85
        timeliness: 1.0
        validity: 0.95
        uniqueness: 0.99
        relevance: 0.80
        stability: 0.90
```

### 監視スケジュール (`quality_monitoring.schedule`)

```yaml
quality_monitoring:
  schedule:
    # 監視頻度
    monitoring_frequency: "hourly" # 'hourly', 'daily', 'weekly'

    # バッチ設定
    batch_size: 1000
    parallel_processing: true

    # レポート設定
    reporting:
      daily_report: true
      weekly_summary: true
      monthly_dashboard: true

      # レポート形式
      report_formats: ["html", "pdf", "json"]

      # 配信設定
      email_recipients: ["admin@example.com"]
      slack_webhook: "https://hooks.slack.com/..."

    # データ保持
    data_retention:
      metrics_retention_days: 90
      alerts_retention_days: 365
      reports_retention_days: 180
```

## A/B テスト設定

### テスト設定 (`ab_testing`)

```yaml
ab_testing:
  # 基本設定
  enabled: true

  # デフォルトテスト設定
  default_config:
    test_type: "feature_comparison"
    min_sample_size: 1000
    power: 0.8
    alpha: 0.05

    # 評価メトリクス
    evaluation_metrics:
      - "accuracy"
      - "precision"
      - "recall"
      - "f1_score"
      - "auc_roc"

    # 統計検定
    statistical_tests:
      - "t_test"
      - "mann_whitney"
      - "chi_square"
      - "fisher_exact"

    # 実験設計
    experimental_design:
      randomization: "simple" # 'simple', 'stratified', 'block'
      control_group_ratio: 0.5
      minimum_effect_size: 0.05

    # 停止条件
    stopping_criteria:
      max_duration_days: 30
      statistical_significance: true
      practical_significance: true
      sample_size_reached: true

  # 高度な設定
  advanced_settings:
    # 多重比較補正
    multiple_comparisons_correction: "bonferroni"

    # シーケンシャルテスト
    sequential_testing: false

    # ベイジアンA/Bテスト
    bayesian_testing: false
    bayesian_prior: "uniform"

    # セグメンテーション分析
    segmentation_analysis: true
    segmentation_variables: ["user_type", "region", "device"]
```

## 環境設定

### システム設定 (`environment.system`)

```yaml
environment:
  system:
    # Python環境
    python_version: "3.8+"
    virtual_env: true

    # 依存関係
    dependencies:
      numpy: ">=1.19.0"
      pandas: ">=1.3.0"
      scikit-learn: ">=1.0.0"
      torch: ">=1.9.0"
      torch-geometric: ">=2.0.0"

    # システムリソース
    resources:
      max_memory_gb: 8
      max_cpu_cores: 4
      gpu_enabled: false

    # ファイルシステム
    filesystem:
      temp_dir: "/tmp/feature_pipeline"
      output_dir: "./outputs"
      log_dir: "./logs"
      cache_dir: "./cache"

      # ディスク容量制限
      max_temp_size_gb: 5
      max_output_size_gb: 10
      max_cache_size_gb: 2
```

### セキュリティ設定 (`environment.security`)

```yaml
environment:
  security:
    # データ保護
    data_encryption: false
    encryption_algorithm: "AES-256"

    # アクセス制御
    access_control: false
    api_key_required: false

    # 監査ログ
    audit_logging: true
    audit_log_file: "audit.log"

    # 設定の検証
    config_validation: true
    schema_validation: true
```

## 設定例

### 基本設定例

```yaml
# basic_config.yaml - 基本的な設定例
pipeline:
  stages: ["analysis", "design", "optimization"]
  enable_cache: true
  cache_expiry_hours: 12
  error_handling: "continue"
  log_level: "INFO"

analysis:
  importance_analysis:
    enabled: true
    analysis_types: ["absolute", "relative"]
    statistical_test: true

  correlation_analysis:
    enabled: true
    correlation_methods: ["pearson"]
    high_correlation_threshold: 0.8

design:
  task_features:
    enabled: true
    feature_types: ["urgency", "complexity"]
    enhancement_methods: ["nlp_analysis"]

  developer_features:
    enabled: true
    feature_types: ["expertise", "activity_pattern"]

optimization:
  scaling:
    enabled: true
    methods: ["standard", "robust"]
    auto_selection: true

  selection:
    enabled: true
    methods: ["rfe", "lasso"]
    target_feature_count: 50
```

### 高度な設定例

```yaml
# advanced_config.yaml - 高度な設定例
pipeline:
  stages:
    ["analysis", "design", "optimization", "gat_enhancement", "evaluation"]
  enable_cache: true
  cache_expiry_hours: 24
  enable_parallel: true
  max_workers: 8
  max_memory_mb: 4096

analysis:
  importance_analysis:
    enabled: true
    analysis_types: ["absolute", "relative", "category_wise", "temporal"]
    statistical_test: true
    significance_level: 0.01
    multiple_testing_correction: "fdr_bh"
    top_k_features: 100

  correlation_analysis:
    enabled: true
    correlation_methods: ["pearson", "spearman", "kendall"]
    high_correlation_threshold: 0.85
    cluster_correlated_features: true
    clustering_method: "hierarchical"

  distribution_analysis:
    enabled: true
    distribution_tests: ["normality", "skewness", "kurtosis", "outliers"]
    outlier_detection_methods: ["iqr", "zscore", "isolation_forest"]

design:
  task_features:
    enabled: true
    feature_types:
      ["urgency", "complexity", "social_attention", "temporal", "semantic"]
    enhancement_methods:
      ["nlp_analysis", "time_series", "graph_analysis", "clustering"]

    nlp_config:
      embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
      max_sequence_length: 512

    urgency_calculation:
      priority_weights:
        critical: 4.0
        high: 3.0
        medium: 2.0
        low: 1.0
      deadline_pressure_window_days: 7

  developer_features:
    enabled: true
    feature_types:
      ["expertise", "activity_pattern", "quality", "collaboration", "learning"]
    temporal_window_days: 120

    expertise_calculation:
      min_commits_for_expertise: 30
      expertise_decay_rate: 0.08
      diversity_bonus_factor: 1.3

  matching_features:
    enabled: true
    matching_types:
      [
        "temporal_proximity",
        "technical_compatibility",
        "success_history",
        "workload_balance",
      ]
    historical_data_months: 18

optimization:
  scaling:
    enabled: true
    methods: ["standard", "minmax", "robust", "quantile"]
    auto_selection: true

    feature_type_mapping:
      continuous: "standard"
      discrete: "robust"
      categorical: "onehot"

  selection:
    enabled: true
    methods: ["univariate", "rfe", "lasso", "mutual_info", "stability"]
    target_feature_count: 75

    stability_selection:
      n_bootstrap_iterations: 100
      threshold: 0.7

    evaluation_metric: "f1_score"
    cross_validation: true
    cv_folds: 5

  dimension_reduction:
    enabled: true
    methods: ["pca", "umap"]
    target_dimensions: 64
    preserve_variance_ratio: 0.98

gat_enhancement:
  optimization:
    enabled: true
    dimension_candidates: [32, 64, 128, 256]
    evaluation_metrics:
      ["embedding_quality", "downstream_performance", "attention_diversity"]

    gat_architecture:
      num_layers: 3
      hidden_dim: 128
      num_heads: 8
      dropout: 0.1

    training:
      learning_rate: 0.001
      max_epochs: 300
      early_stopping: true
      patience: 30

  interpretation:
    enabled: true
    interpretation_methods: ["attention_analysis", "embedding_analysis"]

    visualization:
      network_layout: "spring"
      figure_size: [16, 12]
      save_format: "png"

  integration:
    enabled: true
    integration_strategies:
      ["concatenation", "weighted_sum", "attention_fusion"]

    redundancy_removal:
      correlation_threshold: 0.85
      importance_threshold: 0.005

quality_monitoring:
  metrics:
    enabled: true
    quality_metrics: ["completeness", "consistency", "accuracy", "stability"]

    alert_thresholds:
      completeness: 0.98
      consistency: 0.95
      accuracy: 0.90
      stability: 0.95

  schedule:
    monitoring_frequency: "daily"
    daily_report: true
    weekly_summary: true

ab_testing:
  enabled: true
  default_config:
    min_sample_size: 2000
    power: 0.9
    alpha: 0.01
    evaluation_metrics:
      ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    statistical_tests: ["t_test", "mann_whitney"]

  advanced_settings:
    multiple_comparisons_correction: "fdr_bh"
    sequential_testing: true
    segmentation_analysis: true

environment:
  system:
    max_memory_gb: 16
    max_cpu_cores: 8
    gpu_enabled: true

  security:
    audit_logging: true
    config_validation: true
```

### プロダクション設定例

```yaml
# production_config.yaml - プロダクション環境設定例
pipeline:
  stages:
    ["analysis", "design", "optimization", "gat_enhancement", "evaluation"]
  enable_cache: true
  cache_expiry_hours: 6 # 頻繁な更新
  error_handling: "continue"
  max_retries: 5
  enable_parallel: true
  max_workers: 16
  max_memory_mb: 8192
  log_level: "WARNING" # エラーのみログ

# 重要な分析のみ有効化
analysis:
  importance_analysis:
    enabled: true
    analysis_types: ["absolute", "relative"]
    statistical_test: false # 高速化のため

  correlation_analysis:
    enabled: true
    correlation_methods: ["pearson"] # 最も高速な手法のみ
    high_correlation_threshold: 0.9 # 厳しい閾値

# 効率的な特徴量設計
design:
  task_features:
    enabled: true
    feature_types: ["urgency", "complexity"] # 重要な特徴量のみ
    enhancement_methods: ["nlp_analysis"]

  developer_features:
    enabled: true
    feature_types: ["expertise", "quality"]
    temporal_window_days: 60 # 短期間

# 高速最適化
optimization:
  scaling:
    enabled: true
    methods: ["standard"] # 最も高速
    auto_selection: false

  selection:
    enabled: true
    methods: ["univariate"] # 最も高速
    target_feature_count: 50
    cross_validation: false # 高速化

# GAT最適化を簡素化
gat_enhancement:
  optimization:
    enabled: true
    dimension_candidates: [64, 128] # 限定的な探索
    evaluation_metrics: ["downstream_performance"] # 重要なメトリクスのみ

  interpretation:
    enabled: false # プロダクションでは不要

  integration:
    enabled: true
    integration_strategies: ["concatenation"] # 最もシンプル

# プロダクション監視
quality_monitoring:
  metrics:
    enabled: true
    quality_metrics: ["completeness", "accuracy"] # 重要なメトリクスのみ

  schedule:
    monitoring_frequency: "hourly"
    daily_report: true

    # アラート設定を厳格化
    alerts:
      enable_alerts: true
      alert_channels: ["email", "slack"]
      alert_thresholds:
        completeness: 0.99
        accuracy: 0.95

# A/Bテストは無効化（プロダクションでは別途実行）
ab_testing:
  enabled: false

# プロダクション環境設定
environment:
  system:
    max_memory_gb: 32
    max_cpu_cores: 16
    gpu_enabled: true

  security:
    data_encryption: true
    access_control: true
    audit_logging: true
    config_validation: true
```

この設定ファイルリファレンスは、IRL 特徴量リデザインシステムのすべての設定オプションの詳細な説明を提供します。実際の使用では、プロジェクトの要件に応じて適切な設定を選択してください。
