# 設定ファイル移行ガイド

## 🔄 リファクタリング完了！

設定ファイルが以下の構造に整理されました：

### 📁 新しいディレクトリ構造

```
configs/
├── base/                     # 基本設定
│   ├── common.yaml          # 共通設定
│   ├── data_paths.yaml      # データパス設定
│   ├── irl_settings.yaml    # IRL設定集
│   └── rl_settings.yaml     # RL設定集
├── environments/            # 環境別設定
│   ├── development.yaml     # 開発環境
│   ├── production.yaml      # 本番環境
│   └── testing.yaml         # テスト環境
├── training/               # トレーニング設定
│   ├── base_training.yaml   # 基本トレーニング
│   ├── improved_training.yaml # 改良トレーニング
│   └── multi_method_training.yaml # マルチメソッド
├── evaluation/             # 評価設定
│   ├── base_evaluation.yaml # 基本評価
│   └── hybrid_evaluation.yaml # ハイブリッド評価
├── data/                   # データファイル
│   ├── dev_profiles_2022.yaml
│   ├── dev_profiles_2023.yaml
│   └── dev_profiles_unified.yaml
└── old_configs/            # バックアップ
```

### 🔄 旧設定から新設定への移行マッピング

| 旧設定ファイル | 新設定ファイル | 用途 |
|---|---|---|
| `rl_debug.yaml` | `environments/development.yaml` | 開発・デバッグ |
| `simple_test.yaml` | `environments/development.yaml` | 開発・デバッグ |
| `production.yaml` | `environments/production.yaml` | 本番環境 |
| `base_training.yaml` | `training/base_training.yaml` | 基本トレーニング |
| `improved_ppo_training.yaml` | `training/improved_training.yaml` | 改良トレーニング |
| `multi_method_training.yaml` | `training/multi_method_training.yaml` | マルチメソッド |
| `base_test_2022.yaml` | `evaluation/base_evaluation.yaml` | 基本評価 |

### 🚀 使用方法

#### 1. 設定ローダーを使用（推奨）

```python
from utils.config_loader import load_config

# 新しい設定システムを使用
config = load_config('environments/development.yaml')
```

#### 2. 従来の方法（互換性維持）

```python
import yaml
from utils.config_loader import SimpleConfig

with open('configs/environments/development.yaml') as f:
    config_dict = yaml.safe_load(f)
config = SimpleConfig(config_dict)
```

### ✨ 新機能

1. **階層的設定継承**: `base_configs`で基本設定を自動継承
2. **設定参照**: データパスやコンポーネント設定を名前で参照
3. **環境別設定**: development/production/testingの明確な分離
4. **設定検証**: 設定の整合性チェック機能

### 🔧 既存コードの修正例

#### Before:
```python
config_path = "configs/rl_debug.yaml"
```

#### After:
```python
config_path = "environments/development.yaml"
```

### 💡 Tips

- 開発時は `environments/development.yaml` を使用
- 本番環境では `environments/production.yaml` を使用
- 新しいトレーニング実験は `training/` 下の設定を参考に作成
- カスタム設定は既存の設定を継承して作成可能

### ❓ トラブルシューティング

設定が見つからない場合：
1. `old_configs/` ディレクトリで旧設定を確認
2. 上記のマッピング表で新しいパスを確認
3. `utils/config_loader.py` のテスト機能で動作確認

```bash
cd utils && python config_loader.py
```
