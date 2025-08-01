#!/usr/bin/env python3
"""
設定ファイルマイグレーションスクリプト
既存の古い設定ファイルから新しいリファクタリングされた設定への移行をサポート
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigMigrator:
    """設定ファイルマイグレーター"""

    def __init__(self, config_root: str):
        self.config_root = Path(config_root)
        self.old_configs_dir = self.config_root / "old_configs"

    def create_backup(self):
        """既存の設定ファイルをバックアップ"""
        print("📦 既存設定ファイルをバックアップ中...")

        if not self.old_configs_dir.exists():
            self.old_configs_dir.mkdir()

        # 古いファイルをバックアップディレクトリに移動
        old_files = [
            "base_test_2022.yaml",
            "base_test_2023.yaml",
            "base_training.yaml",
            "base_training_2022.yaml",
            "dev_profiles_training.yaml",
            "dev_profiles_training_2022.yaml",
            "improved_ppo_training.yaml",
            "improved_rl_training.yaml",
            "multi_method_training.yaml",
            "production.yaml",
            "rl_debug.yaml",
            "rl_experiment.yaml",
            "simple_test.yaml",
            "unified_rl.yaml",
        ]

        backed_up = 0
        for filename in old_files:
            old_path = self.config_root / filename
            if old_path.exists():
                backup_path = self.old_configs_dir / filename
                shutil.move(str(old_path), str(backup_path))
                backed_up += 1
                print(f"  ✅ {filename} -> old_configs/")

        print(f"📦 {backed_up}個のファイルをバックアップしました")

    def create_migration_mapping(self) -> Dict[str, str]:
        """旧設定から新設定へのマッピングを作成"""
        return {
            # 開発・デバッグ用
            "rl_debug.yaml": "environments/development.yaml",
            "simple_test.yaml": "environments/development.yaml",
            # 本番用
            "production.yaml": "environments/production.yaml",
            # トレーニング用
            "base_training.yaml": "training/base_training.yaml",
            "base_training_2022.yaml": "training/base_training.yaml",
            "improved_ppo_training.yaml": "training/improved_training.yaml",
            "improved_rl_training.yaml": "training/improved_training.yaml",
            "multi_method_training.yaml": "training/multi_method_training.yaml",
            "unified_rl.yaml": "training/multi_method_training.yaml",
            # 評価用
            "base_test_2022.yaml": "evaluation/base_evaluation.yaml",
            "base_test_2023.yaml": "evaluation/base_evaluation.yaml",
        }

    def generate_usage_guide(self):
        """使用ガイドを生成"""
        guide_content = """# 設定ファイル移行ガイド

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
"""

        guide_path = self.config_root / "MIGRATION_GUIDE.md"
        with open(guide_path, "w", encoding="utf-8") as f:
            f.write(guide_content)

        print(f"📖 使用ガイドを作成: {guide_path}")

    def update_hybrid_recommendation_system(self):
        """hybrid_recommendation_system.pyを新しい設定システムに対応"""
        hybrid_file = (
            self.config_root.parent / "evaluation" / "hybrid_recommendation_system.py"
        )

        if not hybrid_file.exists():
            print("⚠️  hybrid_recommendation_system.py が見つかりません")
            return

        print("🔄 hybrid_recommendation_system.py を更新中...")

        # ConfigLoaderのインポートを追加
        import_addition = """
# 新しい設定システムを使用
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from config_loader import load_config"""

        # SimpleConfigクラス定義を削除してconfig_loaderから使用
        old_simple_config = '''class SimpleConfig:
    """辞書をオブジェクトのように扱うためのクラス"""

    def __init__(self, config_dict):
        self._dict = config_dict
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, SimpleConfig(value))
            else:
                setattr(self, key, value)

    def get(self, key, default=None):
        return self._dict.get(key, default)'''

        new_simple_config = """# SimpleConfigは config_loader から使用"""

        # 設定読み込み部分を更新
        old_config_load = """    # 設定読み込み
    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config = SimpleConfig(config_dict)"""

        new_config_load = """    # 設定読み込み（新しいシステム）
    try:
        config = load_config(args.config)
        print(f"✅ 新しい設定システムで読み込み: {args.config}")
    except Exception as e:
        print(f"⚠️  新しい設定システムで失敗、従来方式で読み込み: {e}")
        with open(args.config, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        from config_loader import SimpleConfig
        config = SimpleConfig(config_dict)"""

        try:
            with open(hybrid_file, "r", encoding="utf-8") as f:
                content = f.read()

            # 更新を適用
            if "from config_loader import" not in content:
                # インポートを追加
                content = content.replace(
                    'sys.path.append(str(Path(__file__).parent.parent / "src"))',
                    'sys.path.append(str(Path(__file__).parent.parent / "src"))'
                    + import_addition,
                )

            # SimpleConfigクラス定義を置換
            if old_simple_config in content:
                content = content.replace(old_simple_config, new_simple_config)

            # 設定読み込み部分を置換
            if old_config_load in content:
                content = content.replace(old_config_load, new_config_load)

            # ファイルを更新
            with open(hybrid_file, "w", encoding="utf-8") as f:
                f.write(content)

            print("✅ hybrid_recommendation_system.py を更新しました")

        except Exception as e:
            print(f"❌ ファイル更新エラー: {e}")

    def run_migration(self):
        """完全マイグレーションを実行"""
        print("🚀 設定ファイルマイグレーション開始")
        print("=" * 50)

        # 1. バックアップ作成
        self.create_backup()
        print()

        # 2. 使用ガイド生成
        self.generate_usage_guide()
        print()

        # 3. hybrid_recommendation_system.py更新
        self.update_hybrid_recommendation_system()
        print()

        print("✅ マイグレーション完了！")
        print("📖 詳細は MIGRATION_GUIDE.md を参照してください")


def main():
    parser = argparse.ArgumentParser(description="設定ファイルマイグレーション")
    parser.add_argument(
        "--config-root", default=".", help="設定ファイルのルートディレクトリ"
    )
    parser.add_argument(
        "--backup-only", action="store_true", help="バックアップのみ実行"
    )

    args = parser.parse_args()

    migrator = ConfigMigrator(args.config_root)

    if args.backup_only:
        migrator.create_backup()
    else:
        migrator.run_migration()


if __name__ == "__main__":
    main()
