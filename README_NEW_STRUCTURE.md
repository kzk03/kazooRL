# Kazoo プロジェクト - 新構造

## 📁 ディレクトリ構造

```
kazoo/
├── training/           # 学習関連スクリプト
│   ├── gat/           # GAT (Graph Attention Network) 訓練
│   ├── irl/           # IRL (Inverse Reinforcement Learning) 訓練  
│   └── rl/            # RL (Reinforcement Learning) 訓練
├── pipelines/         # 統合パイプライン
├── analysis/          # 分析・レポート
│   ├── reports/       # 分析レポート生成
│   └── visualization/ # 可視化
├── evaluation/        # 評価・テスト
├── data_processing/   # データ処理
├── utils/             # ユーティリティ
├── src/              # ライブラリコード
├── configs/          # 設定ファイル
├── data/             # データファイル
├── models/           # 学習済みモデル
└── outputs/          # 出力ファイル
```

## 🚀 使用方法

### 完全パイプライン実行
```bash
python pipelines/full_pipeline.py
```

### 個別ステップ実行

#### GAT訓練
```bash
python training/gat/train_gat.py
```

#### IRL訓練  
```bash
python training/irl/train_irl.py
```

#### RL訓練
```bash
python training/rl/train_rl.py
```

### 分析・レポート生成
```bash
python analysis/reports/irl_analysis.py
```

### 評価
```bash
python evaluation/evaluate_models.py
```

## 📊 主要な改善点

- **機能別整理**: GAT、IRL、RLの訓練スクリプトを分離
- **統合分析**: 複数の分析機能を統合し、分かりやすいレポートを生成
- **パイプライン化**: 完全な学習フローを自動実行
- **エラーハンドリング**: 各ステップでのエラー処理を強化

## 🔧 設定

主要な設定は `configs/base_training.yaml` で管理されています。

## 📈 出力

- 学習済みモデル: `models/`
- 分析結果: `outputs/`
- ログ: `logs/`
