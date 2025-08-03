# 特徴量設計モジュール

IRL 特徴量の設計・改良のための基盤クラスを提供します。

## 概要

このモジュールは、既存特徴量の改良と新規特徴量の追加を行う 3 つの主要クラスを実装しています：

1. **TaskFeatureDesigner**: タスク特徴量設計器
2. **DeveloperFeatureDesigner**: 開発者特徴量設計器
3. **MatchingFeatureDesigner**: マッチング特徴量設計器

## 実装済み機能

### 1. TaskFeatureDesigner

既存タスク特徴量の改良と新規特徴量の追加を行います。

**改良された既存特徴量:**

- テキスト長の対数変換（外れ値の影響を軽減）
- コメント数の対数変換
- 活動度の改良（最終更新からの経過時間）
- タスク年齢（作成からの経過時間）
- コードブロック密度の改良

**新規特徴量:**

**緊急度特徴量:**

- 優先度ラベルベースの緊急度スコア
- 緊急度キーワードの出現頻度・密度
- 期限・マイルストーン関連の言及数
- ブロッキング課題の特徴量

**複雑度特徴量:**

- 技術用語密度
- 参照リンク数（URL、GitHub issue 参照、PR 参照）
- 推定工数スコア（テキスト複雑度ベース）
- 依存関係数の推定
- コード複雑度指標（関数定義、クラス定義）

**社会的注目度特徴量:**

- ウォッチャー数・リアクション数
- メンション数・外部参照数
- 参加者数の推定
- 社会的注目度総合スコア

### 2. DeveloperFeatureDesigner

既存開発者特徴量の改良と新規特徴量の追加を行います。

**改良された既存特徴量:**

- 活動量の対数変換（コミット、PR、Issue、変更行数）
- 協力ネットワークの対数変換
- 効率性指標（比率計算）
- 社会性指標・活動多様性

**新規特徴量:**

**専門性特徴量:**

- 主要言語強度・言語多様性
- ドメイン専門性（ファイルパスから推定）
- 技術多様性・フレームワーク経験
- 学習速度・現代技術への適応度

**活動パターン特徴量:**

- タイムゾーン推定・ピーク活動時間
- 平日・週末活動パターン
- 応答時間パターン・一貫性スコア
- 夜間活動比率・営業時間活動比率

**品質特徴量:**

- PR マージ率・レビュー承認率
- バグ導入率・コード信頼性
- コードレビュー品質・詳細度
- テスト・ドキュメント品質
- 総合品質スコア

### 3. MatchingFeatureDesigner

既存マッチング特徴量の改良と新規特徴量の追加を行います。

**改良された既存特徴量:**

- 協力強度の改良（頻度と最近性を考慮）
- スキル適合性の改良（完全一致・部分一致・重み付き）
- ファイル専門性関連度の改良（直接・ディレクトリ・ファイルタイプ）

**新規特徴量:**

**時間的近接性特徴量:**

- 最近の協力日数・協力回数
- 活動時間重複・ピーク活動時間の近さ
- タイムゾーン適合性
- 応答時間予測・適合性

**技術的適合性特徴量:**

- 技術スタック重複（言語、フレームワーク、データベース等）
- 言語習熟度マッチング
- フレームワーク経験適合性
- アーキテクチャ親和性
- 技術的複雑度適合性

**成功履歴特徴量:**

- 過去成功率・最近の成功率
- 類似タスク完了率
- 協力満足度・品質履歴
- 納期遵守率
- 成功確率推定・リスク評価

## ファイル構成

```
kazoo/analysis/feature_design/
├── __init__.py                           # モジュール初期化
├── task_feature_designer.py             # タスク特徴量設計器
├── developer_feature_designer.py        # 開発者特徴量設計器
├── matching_feature_designer.py         # マッチング特徴量設計器
├── test_feature_design.py               # 実装確認テスト
└── README.md                             # このファイル
```

## 使用方法

### 基本的な使用例

```python
from task_feature_designer import TaskFeatureDesigner
from developer_feature_designer import DeveloperFeatureDesigner
from matching_feature_designer import MatchingFeatureDesigner

# 1. タスク特徴量設計
task_designer = TaskFeatureDesigner()
task_data = {
    'title': 'Fix API bug',
    'body': 'Critical bug in authentication API',
    'labels': ['critical', 'bug', 'api'],
    'comments': 10,
    'watchers': 15,
    'reactions': {'+1': 5, 'heart': 2}
}

task_features = task_designer.design_enhanced_features(task_data)

# 2. 開発者特徴量設計
dev_designer = DeveloperFeatureDesigner()
developer_data = {
    'total_commits': 1000,
    'languages': {'python': 70, 'javascript': 30},
    'skills': ['python', 'django', 'react'],
    'activity_hours': [9, 10, 14, 15, 16],
    'total_prs': 150,
    'merged_prs': 140
}

dev_features = dev_designer.design_enhanced_features(developer_data)

# 3. マッチング特徴量設計
matching_designer = MatchingFeatureDesigner()
env_context = {
    'collaboration_history': [...],
    'task_history': [...],
    'collaboration_feedback': {...}
}

matching_features = matching_designer.design_enhanced_features(
    task_data, developer_data, env_context
)

# 4. 統合特徴量
all_features = {}
all_features.update(task_features)
all_features.update(dev_features)
all_features.update(matching_features)
```

### 設定のカスタマイズ

```python
# カスタム設定でTaskFeatureDesignerを初期化
custom_config = {
    'priority_labels': {
        'critical': 5, 'high': 4, 'medium': 3, 'low': 2, 'trivial': 1
    },
    'technical_terms': ['API', 'database', 'algorithm', 'security'],
    'urgency_keywords': ['urgent', 'critical', 'asap', 'emergency']
}

task_designer = TaskFeatureDesigner(config=custom_config)
```

## テスト

実装の動作確認：

```bash
cd kazoo/analysis/feature_design
uv run python test_feature_design.py
```

## 出力例

### 実装確認結果

```
🚀 特徴量設計実装確認開始
============================================================

📋 TaskFeatureDesigner テスト
✅ タスク特徴量設計完了: 40個の特徴量
   主要特徴量:
     task_priority_label_score: 5.000
     task_urgency_keyword_count: 4.000
     task_technical_term_density: 7.905
     task_social_attention_score: 81.500

👨‍💻 DeveloperFeatureDesigner テスト
✅ 開発者特徴量設計完了: 61個の特徴量
   主要特徴量:
     dev_primary_language_strength: 0.700
     dev_pr_merge_rate: 0.900
     dev_overall_quality_score: 0.892

🔗 MatchingFeatureDesigner テスト
✅ マッチング特徴量設計完了: 42個の特徴量
   主要特徴量:
     match_skill_compatibility_weighted: 1.000
     match_estimated_success_probability: 0.821
     match_success_confidence: 0.560

📊 特徴量統計:
   - タスク特徴量: 40個
   - 開発者特徴量: 61個
   - マッチング特徴量: 42個
   - 総特徴量数: 143個
```

## 特徴量カテゴリ

### タスク特徴量（40 個）

- **既存改良**: 8 個（対数変換、正規化、密度計算）
- **緊急度**: 7 個（優先度、キーワード、期限、ブロッキング）
- **複雑度**: 10 個（技術用語、参照、工数、依存関係、コード）
- **社会的注目度**: 15 個（ウォッチャー、リアクション、参加者）

### 開発者特徴量（61 個）

- **既存改良**: 12 個（対数変換、比率、効率性、多様性）
- **専門性**: 20 個（言語、ドメイン、技術、学習速度）
- **活動パターン**: 15 個（時間、応答、一貫性）
- **品質**: 14 個（マージ率、レビュー、バグ、テスト）

### マッチング特徴量（42 個）

- **既存改良**: 8 個（協力強度、スキル適合性、ファイル関連度）
- **時間的近接性**: 10 個（協力日数、活動時間、タイムゾーン、応答時間）
- **技術的適合性**: 12 個（技術スタック、言語習熟度、アーキテクチャ）
- **成功履歴**: 12 個（成功率、満足度、品質、リスク評価）

## 要件対応

この実装は以下の要件を満たしています：

- **要件 2.1**: TaskFeatureDesigner クラスの実装 ✅

  - 既存タスク特徴量の改良（対数変換、正規化、複雑度計算）
  - 緊急度特徴量（優先度ラベル、期限、マイルストーン、ブロッキング課題）
  - 複雑度特徴量（技術用語密度、参照リンク、推定工数、依存関係）
  - 社会的注目度特徴量（ウォッチャー、リアクション、メンション、外部参照）

- **要件 2.2**: DeveloperFeatureDesigner クラスの実装 ✅

  - 既存開発者特徴量の改良（対数変換、比率計算、効率性指標）
  - 専門性特徴量（主要言語強度、ドメイン専門性、技術多様性、学習速度）
  - 活動パターン特徴量（タイムゾーン、平日活動比、応答時間、一貫性）
  - 品質特徴量（PR マージ率、レビュー承認率、バグ導入率、コードレビュー品質）

- **要件 2.3**: MatchingFeatureDesigner クラスの実装 ✅
  - 既存マッチング特徴量の改良（協力強度、スキル適合性、ファイル専門性）
  - 時間的近接性特徴量（最近の協力日数、活動時間重複、タイムゾーン適合性）
  - 技術的適合性特徴量（技術スタック重複、言語習熟度、フレームワーク経験）
  - 成功履歴特徴量（過去成功率、類似タスク完了率、協力満足度、成功確率推定）

## 今後の拡張

この特徴量設計基盤は、今後の特徴量最適化・GAT 特徴量最適化タスクの基礎として使用されます：

- タスク 3: 特徴量最適化システムの構築
- タスク 4: GAT 特徴量最適化の実装
- タスク 5: 特徴量パイプラインの自動化
