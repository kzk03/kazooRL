# Kazoo - OSS 開発プロセス マルチエージェント強化学習システム

Kazoo は、オープンソースソフトウェア（OSS）開発プロセスにおけるタスク推薦システムをマルチエージェント強化学習で実現するプロジェクトです。Graph Attention Network（GAT）特徴量、逆強化学習（IRL）、PPO 強化学習を組み合わせて、開発者への最適なタスク推薦を行います。

## 🌟 主な特徴

- **GAT 特徴量**: Graph Attention Network による開発者・タスク間の関係性学習
- **逆強化学習**: エキスパート軌跡からの報酬関数学習
- **マルチエージェント強化学習**: 複数開発者の協調学習
- **オンライン学習**: リアルタイムでの学習更新
- **リアルタイム進捗表示**: 色分けされた tqdm 進捗バーによる可視化

## 🚀 クイックスタート

### 1. 環境構築

```bash
# プロジェクトのクローン
git clone <repository-url>
cd kazoo

# 依存関係のインストール（uv使用）
uv sync
```

### 2. データ準備

```bash
# 学習・テスト用データの生成
python tools/data_processing/generate_backlog.py
python tools/data_processing/generate_profiles.py
```

### 3. 統合学習パイプラインの実行

```bash
# フル学習パイプライン（GAT → IRL → RL）
python scripts/full_training_pipeline.py --config configs/base_training.yaml

# 個別ステップの実行
python scripts/full_training_pipeline.py --skip-gnn    # GNN訓練をスキップ
python scripts/full_training_pipeline.py --skip-irl    # IRL訓練をスキップ
python scripts/full_training_pipeline.py --skip-rl     # RL訓練をスキップ
```

### 4. 評価

```bash
# 2022年テストデータでの評価
python scripts/evaluate_2022_test.py \
  --config configs/base_test_2022.yaml \
  --learned-weights data/learned_weights_training.npy
```

## 📊 進捗表示

学習中は色分けされた進捗バーで進行状況を可視化：

- 🧠 **GAT 訓練** (シアン): エポック進行、損失統計
- 🎯 **IRL 学習** (青): エポック・ステップ進行、平均損失、有効ステップ数
- 🤖 **PPO 学習** (マゼンタ): Update・Step 進行、報酬統計
- 📊 **評価** (緑): タスク進行、Top-K 精度リアルタイム表示

## 🏗️ アーキテクチャ

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GAT Training  │───▶│   IRL Learning  │───▶│   RL Training   │
│  (Graph-based   │    │ (Reward from    │    │ (Multi-agent    │
│   features)     │    │  expert data)   │    │   learning)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Developer-Task  │    │ Reward Weights  │    │ Trained Agents  │
│   Embeddings    │    │   (.npy file)   │    │   (.pt files)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 プロジェクト構造

```
kazoo/
├── configs/                    # 設定ファイル
│   ├── base_training.yaml     # 学習用設定
│   └── base_test_2022.yaml    # テスト用設定
├── data/                      # データファイル
│   ├── backlog_training.json  # 学習用バックログ
│   ├── backlog_test_2022.json # テスト用バックログ
│   └── *.pt, *.npy           # 学習済みモデル・重み
├── scripts/                   # 実行スクリプト
│   ├── full_training_pipeline.py  # 統合学習パイプライン
│   ├── train_collaborative_gat.py # GAT訓練
│   ├── evaluate_2022_test.py      # 評価スクリプト
│   └── ...
├── src/kazoo/                 # コアライブラリ
│   ├── agents/               # エージェント実装
│   ├── envs/                 # 環境定義
│   ├── features/             # 特徴量抽出
│   ├── gnn/                  # GNNモデル
│   ├── learners/             # 学習アルゴリズム
│   └── utils/                # ユーティリティ
└── tools/                    # データ処理ツール
    └── data_processing/      # データ生成・変換
```

## ⚙️ 設定

### 主要パラメータ（`configs/base_training.yaml`）

```yaml
num_developers: 5170 # 対象開発者数
irl:
  epochs: 300 # IRL学習エポック数
  learning_rate: 0.0005 # IRL学習率
rl:
  total_timesteps: 2000000 # RL学習ステップ数
  learning_rate: 0.0001 # RL学習率
```

## 📈 評価指標

- **Top-1 Accuracy**: 推薦 1 位が正解の割合
- **Top-3 Accuracy**: 推薦上位 3 位内に正解が含まれる割合
- **Top-5 Accuracy**: 推薦上位 5 位内に正解が含まれる割合

## 🔧 開発

### テスト実行

```bash
# 進捗バーのテスト
python test_progress_bars.py

# 単体テスト
pytest tests/
```

### コード品質

```bash
# Ruffによるリンター・フォーマッター
uv run ruff check .
uv run ruff format .
```

## 📜 ライセンス

このプロジェクトは [ライセンス名] のもとで公開されています。

## 🤝 コントリビューション

プルリクエストや Issue の作成を歓迎します。大きな変更を提案する場合は、まず Issue で議論をお願いします。
