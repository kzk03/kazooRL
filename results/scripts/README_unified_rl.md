# 統合強化学習システム

GNN 特徴量と IRL 重みを統合した RL 訓練システム

## 概要

このシステムは以下の 3 つのアプローチを統合します：

1. **元システム** (`train_oss.py`): `OSSSimpleEnv` + `IndependentPPOController`
2. **直接統合** (`train_rl_agent.py`): カスタム環境 + Stable-Baselines3
3. **統合システム** (`train_unified_rl.py`): 両方のアプローチを組み合わせ

## ファイル構成

```
scripts/
├── train_unified_rl.py      # 統合システムメイン
├── run_unified_rl.py        # 実行ラッパー
├── train_oss.py             # 元システム
└── train_rl_agent.py        # 直接統合版

configs/
└── unified_rl.yaml          # 統合システム設定

outputs/
├── unified_rl_evaluation_*.csv      # 評価結果
├── unified_feature_importance_*.csv # 特徴量重要度
└── feature_distribution_report_*.txt # 分布分析
```

## 使用方法

### 1. 基本実行

```bash
# デフォルト設定で統合訓練
python scripts/run_unified_rl.py

# または直接実行
python scripts/train_unified_rl.py --config-name=unified_rl
```

### 2. 訓練方法の選択

```bash
# 元システムのみ
python scripts/run_unified_rl.py --method original

# Stable-Baselines3のみ
python scripts/run_unified_rl.py --method stable_baselines

# 統合システム（デフォルト）
python scripts/run_unified_rl.py --method unified
```

### 3. クイック実行（テスト用）

```bash
# 高速実行（少ないステップ数）
python scripts/run_unified_rl.py --quick

# 評価のみ
python scripts/run_unified_rl.py --eval-only
```

### 4. パラメータ調整

```bash
# 訓練ステップ数を指定
python scripts/run_unified_rl.py --timesteps 100000

# 設定ファイルを指定
python scripts/run_unified_rl.py --config base_training
```

## 主な機能

### 🎯 IRL 重み統合

- IRL 学習済み重みを報酬関数に直接統合
- 元の環境報酬との重み付き組み合わせ

### 📊 自動評価・レポート生成

- 複数モデルの性能比較
- 特徴量重要度分析
- CSV 形式の結果出力

### 🔧 柔軟な設定管理

- Hydra 設定システム
- コマンドラインでのパラメータオーバーライド
- 計算効率化オプション

### 🚀 自動化パイプライン

- データ読み込み → 訓練 → 評価 → レポート生成
- エラーハンドリング
- 進捗表示

## 設定ファイル (`configs/unified_rl.yaml`)

```yaml
# 訓練方法: original / stable_baselines / unified
training_method: unified

# IRL設定
irl:
  output_weights_path: "data/learned_weights_training.npy"
  irl_weight_factor: 0.5 # IRL報酬の重み

# RL設定
rl:
  total_timesteps: 50000
  learning_rate: 3e-4
  batch_size: 64

# 計算効率化
optimization:
  max_developers: 50
  max_tasks: 200
```

## 出力ファイル

### 評価結果 (`outputs/unified_rl_evaluation_*.csv`)

```csv
model_name,avg_reward,std_reward,max_reward,min_reward,num_episodes
unified_rl_agent,15.23,2.45,18.67,11.89,10
```

### 特徴量重要度 (`outputs/unified_feature_importance_*.csv`)

```csv
feature_name,irl_weight,abs_weight,importance_rank
task_label_question,1.7329,1.7329,1
match_file_experience_count,1.4177,1.4177,2
```

## トラブルシューティング

### 依存関係エラー

```bash
# 必要パッケージをインストール
pip install stable-baselines3[extra] torch pandas numpy
```

### IRL 重みが見つからない

```bash
# 先にIRL訓練を実行
python scripts/train_irl.py
```

### メモリ不足

```bash
# クイックモードで実行
python scripts/run_unified_rl.py --quick
```

## 比較: システム別の特徴

| 項目     | train_oss.py   | train_rl_agent.py | train_unified_rl.py |
| -------- | -------------- | ----------------- | ------------------- |
| 設定管理 | Hydra ✅       | 手動設定          | Hydra ✅            |
| IRL 統合 | 不明確         | 明確 ✅           | 明確 ✅             |
| 環境     | OSSSimpleEnv   | カスタム          | 継承拡張 ✅         |
| 学習器   | IndependentPPO | Stable-Baselines3 | 両方 ✅             |
| 評価     | 基本的         | 基本的            | 包括的 ✅           |
| レポート | なし           | なし              | CSV 出力 ✅         |

## 開発者向け

### 新しい訓練方法の追加

`train_unified_rl.py`の`main()`関数に新しい条件分岐を追加：

```python
elif training_method == 'new_method':
    print("4. 新しい方法で訓練...")
    train_new_method(cfg, env)
```

### カスタム報酬関数

`UnifiedTaskAssignmentEnv.calculate_irl_reward()`を修正：

```python
def calculate_irl_reward(self, task, developer) -> float:
    # カスタム報酬ロジックを実装
    pass
```

## 参考

- [Stable-Baselines3 ドキュメント](https://stable-baselines3.readthedocs.io/)
- [Hydra 設定管理](https://hydra.cc/)
- 元実装: `training/rl/train_oss.py`
