# Kazoo 統合学習パイプライン

## 🚀 クイックスタート

```bash
# フル実行（GNN → IRL → RL）
python scripts/full_training_pipeline.py

# プロダクション設定での実行
python scripts/full_training_pipeline.py --production

# 部分実行
python scripts/full_training_pipeline.py --skip-gnn         # GNNスキップ
python scripts/full_training_pipeline.py --skip-irl         # IRLスキップ
python scripts/full_training_pipeline.py --skip-gnn --skip-irl  # RLのみ

# 静粛モード
python scripts/full_training_pipeline.py --quiet
```

## ⚙️ 設定ファイル

- `configs/base.yaml` - 基本設定（開発・テスト用）
- `configs/production.yaml` - プロダクション設定（サーバー用）

## 📊 出力

- **ログ**: `logs/kazoo_training_YYYYMMDD_HHMMSS.log`
- **レポート**: `logs/kazoo_report_YYYYMMDD_HHMMSS.json`
- **モデル**: `data/`, `models/` ディレクトリ

## ⏱️ 実行時間の目安

| ステップ | 時間            |
| -------- | --------------- |
| GNN 訓練 | 30 分〜1 時間   |
| IRL 学習 | 2〜4 時間       |
| RL 学習  | 8〜12 時間      |
| **合計** | **10〜17 時間** |

## 🔧 オプション

```bash
python scripts/full_training_pipeline.py --help
```

すべてのオプションと詳細説明が表示されます。
