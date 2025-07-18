# 時系列データ分割手順書

2023 年データ追加時の理想的なデータ分割手順

## 📋 概要

データリークを防ぐため、以下の時系列分割を実行します：

- **IRL 学習**: 2019-2021 年 (expert trajectories 用)
- **RL 訓練**: 2022 年 (強化学習訓練用)
- **テスト**: 2023 年 (最終評価用)

## 🛠️ 前提条件

### 必要なファイル

1. `data/backlog_with_2023.json` - 2023 年データを含む統合バックログ
2. 既存の設定ファイル (`configs/unified_rl.yaml`)

### 必要なパッケージ

```bash
pip install click pyyaml
```

## 📝 実行手順

### ステップ 1: 事前確認

```bash
# 現在のデータ状況を確認
python -c "
import json
with open('data/backlog.json', 'r') as f:
    data = json.load(f)
years = {}
for task in data:
    year = task['created_at'][:4]
    years[year] = years.get(year, 0) + 1
for year, count in sorted(years.items()):
    print(f'{year}: {count:,}タスク')
"
```

### ステップ 2: 分割スクリプトの実行

#### 2-1. ドライラン（確認のみ）

```bash
# 分割結果を確認（実際の変更なし）
python scripts/split_temporal_data.py --dry-run --input-file data/backlog_with_2023.json
```

#### 2-2. 実際の分割実行

```bash
# 時系列分割を実行
python scripts/split_temporal_data.py --input-file data/backlog_with_2023.json
```

#### 2-3. カスタム設定での実行

```bash
# 出力先やバックアップ先を指定
python scripts/split_temporal_data.py \
    --input-file data/backlog_with_2023.json \
    --output-dir data/temporal_split_custom \
    --backup-dir backups/my_backup \
    --config-path configs/unified_rl.yaml
```

### ステップ 3: 結果確認

```bash
# 生成されたファイルを確認
ls -la data/temporal_split/
cat data/temporal_split/temporal_split_report.md
```

### ステップ 4: 設定の検証

```bash
# 更新された設定を確認
cat configs/unified_rl.yaml
```

## 📁 生成されるファイル

### データファイル

- `data/temporal_split/backlog_irl_2019_2021.json` - IRL 学習用（2019-2021 年）
- `data/temporal_split/backlog_training_2022.json` - RL 訓練用（2022 年）
- `data/temporal_split/backlog_test_2023.json` - テスト用（2023 年）

### バックアップファイル

- `backups/temporal_split/unified_rl.yaml.backup_YYYYMMDD_HHMMSS`
- `backups/temporal_split/backlog_training.json.backup_YYYYMMDD_HHMMSS`

### レポートファイル

- `data/temporal_split/temporal_split_report.md` - 分割結果の詳細レポート

## ⚙️ 設定の変更内容

分割後の `configs/unified_rl.yaml`:

```yaml
env:
  backlog_path: "data/temporal_split/backlog_training_2022.json" # 2022年データ
  dev_profiles_path: "configs/dev_profiles_training.yaml"
  expert_trajectories_path: "data/expert_trajectories_2019_2021.pkl"

evaluation:
  test_data_path: "data/temporal_split/backlog_test_2023.json" # 2023年データ

# データ分割について:
#   IRL学習期間: 2019-2021年
#   RL訓練期間: 2022年
#   テスト期間: 2023年
#   分割理由: データリーク防止のための時系列分割
```

## 🔄 後続作業

### 1. IRL 学習の再実行

```bash
# 2019-2021年データでexpert trajectoriesを再生成
python scripts/create_expert_trajectories.py --data-path data/temporal_split/backlog_irl_2019_2021.json
```

### 2. RL 訓練の実行

```bash
# 2022年データで強化学習を訓練
python scripts/train_simple_unified_rl.py
```

### 3. 2023 年データでの評価

```bash
# 2023年データで最終評価
python scripts/evaluate_temporal_split.py --test-data data/temporal_split/backlog_test_2023.json
```

## ⚠️ 注意事項

### 重要な変更点

1. **既存モデルの無効化**: 以前の訓練済みモデルは 2019-2021 年データで学習されているため使用不可
2. **IRL 再学習必須**: expert trajectories の再生成が必要
3. **評価結果の非互換性**: 以前の評価結果との直接比較は不可

### データリーク防止の確認

- IRL 学習期間 (2019-2021) < RL 訓練期間 (2022) < テスト期間 (2023)
- 各期間のデータが重複していないことを確認

### バックアップ復旧

元の設定に戻したい場合:

```bash
# バックアップから復旧
cp backups/temporal_split/unified_rl.yaml.backup_* configs/unified_rl.yaml
```

## 🎯 期待される効果

1. **データリーク完全防止**: 時系列順での厳密な分割
2. **現実的な性能評価**: 未来データでの汎化性能測定
3. **研究の信頼性向上**: 学術的に妥当な実験設計

## 📞 トラブルシューティング

### Q: 2023 年データが見つからない

A: `data/backlog_with_2023.json`に 2023 年のタスクが含まれているか確認

### Q: 分割後にエラーが発生

A: バックアップから設定を復旧して再実行

### Q: パフォーマンスが大幅に低下

A: 正常です。データリークがなくなったため、より現実的な性能が表示されています

## 📊 実行例

```bash
$ python scripts/split_temporal_data.py --input-file data/backlog_with_2023.json

🚀 時系列データ分割スクリプト開始
============================================================
📂 データ読み込み: data/backlog_with_2023.json
   総タスク数: 6,543
📅 年別データ分割中...
   年別タスク数:
     2019年: 892タスク
     2020年: 1,245タスク
     2021年: 1,657タスク
     2022年: 1,258タスク
     2023年: 1,491タスク
💾 既存設定をバックアップ中...
📦 分割データセット作成中...
   ✅ IRL用データ: 3,794タスク -> data/temporal_split/backlog_irl_2019_2021.json
   ✅ RL訓練用データ: 1,258タスク -> data/temporal_split/backlog_training_2022.json
   ✅ テスト用データ: 1,491タスク -> data/temporal_split/backlog_test_2023.json
⚙️ 設定ファイル更新中...
📊 移行レポート作成中...

✅ 時系列データ分割完了！
```
