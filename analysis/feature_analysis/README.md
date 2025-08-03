# 特徴量分析基盤

IRL 特徴量の分析・最適化のための基盤クラスを提供します。

## 概要

このモジュールは、逆強化学習（IRL）で使用される特徴量の包括的な分析を行うための 3 つの主要クラスを実装しています：

1. **FeatureImportanceAnalyzer**: 特徴量重要度分析器
2. **FeatureCorrelationAnalyzer**: 特徴量相関分析器
3. **FeatureDistributionAnalyzer**: 特徴量分布分析器

## 実装済み機能

### 1. FeatureImportanceAnalyzer

IRL 学習後の重みファイルを読み込み、特徴量重要度を計算・分析します。

**主要機能:**

- IRL 重みに基づく重要度ランキング
- カテゴリ別重要度比較（タスク、開発者、マッチング、GAT 特徴量）
- 統計的有意性検証（t 検定、正規性検定、ANOVA）
- 基本特徴量と GAT 特徴量の比較分析
- 詳細レポート生成と可視化

**使用例:**

```python
from feature_importance_analyzer import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer("weights.npy", feature_names)
results = analyzer.analyze_importance()

# レポート生成
analyzer.generate_importance_report("importance_report.txt")
analyzer.visualize_importance("outputs/", show_plot=False)
```

### 2. FeatureCorrelationAnalyzer

特徴量間のピアソン相関係数を計算し、相関分析を実行します。

**主要機能:**

- 相関行列の計算
- 高相関ペア（|r| > 0.8）の特定
- 冗長特徴量候補の抽出
- カテゴリ間相関分析
- 相関統計の計算
- 詳細レポート生成と可視化

**使用例:**

```python
from feature_correlation_analyzer import FeatureCorrelationAnalyzer

analyzer = FeatureCorrelationAnalyzer(feature_data, feature_names)
results = analyzer.analyze_correlations(correlation_threshold=0.8)

# 高相関ペアの確認
high_corr_pairs = results['high_correlation_pairs']
redundant_features = results['redundant_features']
```

### 3. FeatureDistributionAnalyzer

各特徴量の分布統計を計算し、データ品質問題を特定します。

**主要機能:**

- 分布統計（平均、分散、歪度、尖度）の計算
- 正規性検定（Shapiro-Wilk、Kolmogorov-Smirnov）
- 外れ値検出（IQR 法、Z-score 法）
- スケール不均衡問題の特定
- データ品質問題の自動検出
- カテゴリ別分布分析

**使用例:**

```python
from feature_distribution_analyzer import FeatureDistributionAnalyzer

analyzer = FeatureDistributionAnalyzer(feature_data, feature_names)
results = analyzer.analyze_distributions()

# データ品質問題の確認
quality_issues = results['data_quality_issues']
scale_imbalance = results['scale_imbalance']
```

## ファイル構成

```
kazoo/analysis/feature_analysis/
├── __init__.py                           # モジュール初期化
├── feature_importance_analyzer.py        # 重要度分析器
├── feature_correlation_analyzer.py       # 相関分析器
├── feature_distribution_analyzer.py      # 分布分析器
├── verify_implementation.py              # 実装確認テスト
├── example_usage.py                      # 使用例デモ
└── README.md                             # このファイル
```

## 依存関係

- numpy >= 1.26
- pandas >= 2.0.0
- scipy >= 1.11.0
- matplotlib >= 3.7.0
- scikit-learn >= 1.7.0

## 使用方法

### 基本的な使用例

```python
import numpy as np
from feature_importance_analyzer import FeatureImportanceAnalyzer
from feature_correlation_analyzer import FeatureCorrelationAnalyzer
from feature_distribution_analyzer import FeatureDistributionAnalyzer

# データ準備
feature_data = np.random.randn(1000, 25)  # サンプルデータ
feature_names = [f"feature_{i}" for i in range(25)]
weights = np.random.randn(25)

# 重みファイル保存
np.save("weights.npy", weights)

# 1. 重要度分析
importance_analyzer = FeatureImportanceAnalyzer("weights.npy", feature_names)
importance_results = importance_analyzer.analyze_importance()

# 2. 相関分析
correlation_analyzer = FeatureCorrelationAnalyzer(feature_data, feature_names)
correlation_results = correlation_analyzer.analyze_correlations()

# 3. 分布分析
distribution_analyzer = FeatureDistributionAnalyzer(feature_data, feature_names)
distribution_results = distribution_analyzer.analyze_distributions()
```

### 実際の IRL データでの使用

```bash
# デモンストレーション実行
cd kazoo/analysis/feature_analysis
uv run python example_usage.py
```

このコマンドで、実際の IRL 重みファイル（`learned_weights_bot_excluded.npy`）を使用した分析デモが実行されます。

## テスト

実装の動作確認：

```bash
cd kazoo/analysis/feature_analysis
uv run python verify_implementation.py
```

## 出力例

### 重要度分析結果

```
🏆 重要度ランキング TOP10:
   1. gat_dev_expertise                   | 重み:-1.25615 | 重要度: 1.25615
   2. feature_0                           | 重み:-0.93067 | 重要度: 0.93067
   3. dev_cross_issue_activity            | 重み:-0.90285 | 重要度: 0.90285
   ...

📋 カテゴリ別重要度:
  タスク特徴量              : 平均重要度 0.37812 (6個)
  開発者特徴量              : 平均重要度 0.43141 (6個)
  GAT特徴量                : 平均重要度 0.55324 (6個)
```

### 相関分析結果

```
📊 相関統計:
   - 平均絶対相関: 0.0258
   - 高相関ペア(|r|≥0.8): 0個
   - 冗長特徴量候補: 0個
```

### 分布分析結果

```
⚖️  スケール不均衡:
   - 標準偏差比（最大/最小）: 806.29
   - 深刻なスケール不均衡: いいえ

🚨 データ品質問題:
   - extreme_skewness: 3個
   - high_outlier_ratio: 7個
```

## 生成されるレポート

各分析器は詳細なテキストレポートと可視化図を生成します：

- `importance_analysis_report.txt`: 重要度分析の詳細レポート
- `correlation_analysis_report.txt`: 相関分析の詳細レポート
- `distribution_analysis_report.txt`: 分布分析の詳細レポート
- `feature_importance_analysis_YYYYMMDD_HHMMSS.png`: 重要度分析の可視化
- `feature_correlation_analysis_YYYYMMDD_HHMMSS.png`: 相関分析の可視化
- `feature_distribution_summary_YYYYMMDD_HHMMSS.png`: 分布分析の可視化

## 要件対応

この実装は以下の要件を満たしています：

- **要件 1.1**: FeatureImportanceAnalyzer クラスの実装 ✅

  - IRL 重みに基づく重要度計算
  - カテゴリ別重要度比較
  - 統計的有意性検証
  - 基本特徴量と GAT 特徴量の比較

- **要件 1.2**: FeatureCorrelationAnalyzer クラスの実装 ✅

  - ピアソン相関係数計算
  - 高相関ペア特定（|r| > 0.8）
  - 冗長特徴量候補抽出
  - 相関分析結果の可視化

- **要件 1.3**: FeatureDistributionAnalyzer クラスの実装 ✅
  - 分布統計計算（平均、分散、歪度、尖度）
  - 正規性検定、外れ値検出
  - スケール不均衡問題の特定
  - 分布分析結果のレポート生成

## 今後の拡張

この基盤クラスは、今後の特徴量設計・最適化タスクの基礎として使用されます：

- タスク 2: 強化された特徴量設計の実装
- タスク 3: 特徴量最適化システムの構築
- タスク 4: GAT 特徴量最適化の実装
- タスク 5: 特徴量パイプラインの自動化
