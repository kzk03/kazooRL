# Kazoo 設定ファイル構造

## 🎯 メイン設定ファイル

### `bot_excluded_production.yaml`

- **用途**: 本格的な訓練用設定
- **特徴**: ボット除外、時系列分割準拠、高精度設定
- **実行時間**: 数時間〜1 日

### `quick_test.yaml`

- **用途**: 開発・デバッグ用設定
- **特徴**: 軽量、高速実行
- **実行時間**: 数分〜30 分

## 📊 開発者プロファイル

### `dev_profiles.yaml`

- **用途**: IRL 用（2019-2021）
- **内容**: 7,010 人の開発者プロファイル

### `dev_profiles_training_2022.yaml`

- **用途**: RL 訓練用（2022）
- **内容**: 2022 年アクティブ開発者

### `dev_profiles_test_2023.yaml`

- **用途**: テスト用（2023）
- **内容**: 2023 年アクティブ開発者

## ⏰ 時系列データ分割

```
2019-2021: IRL訓練データ
    ↓
2022: RL訓練データ
    ↓
2023: テストデータ
```

## 🚀 実行方法

```bash
# 本格訓練
uv run python training/irl/train_irl.py --config configs/bot_excluded_production.yaml

# 高速テスト
uv run python training/irl/train_irl.py --config configs/quick_test.yaml
```

## 📁 アーカイブ

- `archive/`: 古い複雑な設定ファイル群
- `old_configs/`: 以前の設定ファイル
