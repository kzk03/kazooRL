# 設定ファイル使用ガイド

## 設定ファイルの構成

- **`base_training.yaml`**: 学習用設定（デフォルト）
- **`base_test_2022.yaml`**: テスト用設定（2022 年データ）

## 使用方法

### 学習時（デフォルト）

```bash
# 以下は全て base_training.yaml を使用
python scripts/train_irl.py
python scripts/full_training_pipeline.py
python tools/analysis/analyze_weights.py
```

### テスト時

```bash
# base_test_2022.yaml を明示的に指定
python scripts/full_training_pipeline.py --config configs/base_test_2022.yaml
python tools/analysis/analyze_weights.py configs/base_test_2022.yaml
```

## 変更内容

- `base.yaml` を削除し、学習用とテスト用の 2 つに統一
- デフォルトは学習用設定 (`base_training.yaml`) を使用
- テスト時は明示的にテスト用設定を指定

## データセットの違い

### base_training.yaml

- 2022 年以外のデータ（2019-2021, 2023-2024）
- 開発者数: 5,170 人
- データファイル: `backlog_training.json`

### base_test_2022.yaml

- 2022 年のデータのみ
- 開発者数: 21 人
- データファイル: `backlog_test_2022.json`
