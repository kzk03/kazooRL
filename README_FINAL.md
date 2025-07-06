# Kazoo - 強化学習ベースのOSS開発支援システム

## 📁 プロジェクト構造

```
kazoo/
├── src/kazoo/           # メインのソースコード
├── training/            # 各種トレーニングスクリプト
│   ├── gat/            # Graph Attention Network関連
│   ├── irl/            # Inverse Reinforcement Learning関連
│   └── rl/             # Reinforcement Learning関連
├── analysis/            # 分析・レポート生成
│   ├── reports/        # 各種分析レポート
│   └── visualization/  # 可視化スクリプト
├── evaluation/          # モデル評価・テスト
├── data_processing/     # データ前処理
├── pipelines/          # エンドツーエンドパイプライン
├── utils/              # ユーティリティ
├── configs/            # 設定ファイル
├── data/               # データファイル
├── models/             # 保存されたモデル
└── outputs/            # 出力結果

```

## 🚀 クイックスタート

### 1. 環境セットアップ
```bash
python utils/project_setup.py
```

### 2. データ準備
```bash
python data_processing/generate_graph.py
python data_processing/generate_profiles.py
python data_processing/generate_labels.py
```

### 3. 完全なトレーニングパイプライン実行
```bash
python pipelines/full_pipeline.py
```

### 4. 個別コンポーネントのトレーニング

#### GAT (Graph Attention Network)
```bash
python training/gat/train_gat.py
```

#### IRL (Inverse Reinforcement Learning)  
```bash
python training/irl/train_irl.py
```

#### RL (Reinforcement Learning)
```bash
python training/rl/train_rl.py
```

## 📊 分析・レポート

### 包括的な分析レポート
```bash
python analysis/reports/summary_report.py
```

### 個別分析
- **IRL分析**: `python analysis/reports/irl_analysis.py`
- **GAT分析**: `python analysis/reports/gat_analysis.py`
- **協力関係分析**: `python analysis/reports/collaboration_analysis.py`
- **トレーニング結果分析**: `python analysis/reports/training_analysis.py`

### 可視化
```bash
python analysis/visualization/plot_results.py
```

## 🧪 評価・テスト

```bash
python evaluation/evaluate_models.py
python evaluation/test_features.py
```

## 📋 主要機能

- **GAT**: 開発者間の協力関係をモデリング
- **IRL**: 専門家の行動から報酬関数を学習
- **RL**: PPOアルゴリズムによる効果的な開発者推薦
- **分析**: 詳細な重み分析と可視化
- **評価**: 包括的なモデル性能評価

## 🔧 設定

設定ファイルは`configs/`フォルダに配置：
- `base.yaml`: 基本設定
- `dev_profiles.yaml`: 開発者プロファイル

## 📈 改善されたポイント

1. **機能別の明確な分離**: GAT、IRL、RL、分析、データ処理が独立
2. **統一された分析システム**: 全ての分析が`analysis/`フォルダに集約
3. **エンドツーエンドパイプライン**: `pipelines/`で完全自動化
4. **充実した評価システム**: `evaluation/`で包括的テスト
5. **再利用可能なユーティリティ**: `utils/`で共通機能

## 🏗️ 開発者向け

新しい機能の追加時は、適切なフォルダに配置してください：
- トレーニング関連: `training/`
- 分析関連: `analysis/`
- 評価関連: `evaluation/`
- データ処理: `data_processing/`
