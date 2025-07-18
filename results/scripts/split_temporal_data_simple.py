#!/usr/bin/env python3
"""
時系列データ分割スクリプト（シンプル版）
clickを使わずに標準ライブラリのみで実装
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from typing import Dict, List

import yaml


def load_backlog_data(filepath: str) -> List[Dict]:
    """バックログデータを読み込み"""
    print(f"📂 データ読み込み: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"   総タスク数: {len(data):,}")
    return data


def split_data_by_year(data: List[Dict]) -> Dict[str, List[Dict]]:
    """年別にデータを分割"""
    print("📅 年別データ分割中...")
    
    year_data = {
        '2019': [],
        '2020': [],
        '2021': [],
        '2022': [],
        '2023': []
    }
    
    for task in data:
        created_at = task.get('created_at', '')
        if created_at:
            year = created_at[:4]
            if year in year_data:
                year_data[year].append(task)
    
    # 統計表示
    print("   年別タスク数:")
    for year, tasks in year_data.items():
        print(f"     {year}年: {len(tasks):,}タスク")
    
    return year_data


def create_split_datasets(year_data: Dict[str, List[Dict]], output_dir: str):
    """分割データセットを作成"""
    print(f"📦 分割データセット作成中... (出力先: {output_dir})")
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. IRL用データ (2019-2021年)
    irl_data = []
    for year in ['2019', '2020', '2021']:
        irl_data.extend(year_data[year])
    
    irl_path = os.path.join(output_dir, 'backlog_irl_2019_2021.json')
    with open(irl_path, 'w', encoding='utf-8') as f:
        json.dump(irl_data, f, ensure_ascii=False, indent=2)
    print(f"   ✅ IRL用データ: {len(irl_data):,}タスク -> {irl_path}")
    
    # 2. RL訓練用データ (2022年)
    rl_data = year_data['2022']
    rl_path = os.path.join(output_dir, 'backlog_training_2022.json')
    with open(rl_path, 'w', encoding='utf-8') as f:
        json.dump(rl_data, f, ensure_ascii=False, indent=2)
    print(f"   ✅ RL訓練用データ: {len(rl_data):,}タスク -> {rl_path}")
    
    # 3. テスト用データ (2023年)
    test_data = year_data['2023']
    test_path = os.path.join(output_dir, 'backlog_test_2023.json')
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"   ✅ テスト用データ: {len(test_data):,}タスク -> {test_path}")
    
    return {
        'irl': irl_path,
        'training': rl_path,
        'test': test_path
    }


def backup_existing_files(backup_dir: str):
    """既存ファイルをバックアップ"""
    print(f"💾 既存設定をバックアップ中... (バックアップ先: {backup_dir})")
    
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    backup_files = [
        'configs/unified_rl.yaml',
        'data/backlog_training.json'
    ]
    
    for filepath in backup_files:
        if os.path.exists(filepath):
            filename = os.path.basename(filepath)
            backup_path = os.path.join(backup_dir, f"{filename}.backup_{timestamp}")
            shutil.copy2(filepath, backup_path)
            print(f"   ✅ バックアップ: {filepath} -> {backup_path}")


def update_config_file(new_paths: Dict[str, str], config_path: str):
    """設定ファイルを更新"""
    print(f"⚙️ 設定ファイル更新中: {config_path}")
    
    # 既存設定を読み込み
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # データパスを更新
    config['env']['backlog_path'] = new_paths['training']
    
    # 評価用設定を追加
    if 'evaluation' not in config:
        config['evaluation'] = {}
    config['evaluation']['test_data_path'] = new_paths['test']
    
    # 設定を保存
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"   ✅ 設定更新完了")
    print(f"      - 訓練データ: {new_paths['training']}")
    print(f"      - テストデータ: {new_paths['test']}")


def create_migration_report(year_data: Dict[str, List[Dict]], new_paths: Dict[str, str], output_dir: str):
    """移行レポートを作成"""
    report_path = os.path.join(output_dir, 'temporal_split_report.md')
    
    print(f"📊 移行レポート作成中: {report_path}")
    
    total_tasks = sum(len(tasks) for tasks in year_data.values())
    irl_count = len(year_data['2019']) + len(year_data['2020']) + len(year_data['2021'])
    
    report_content = f"""# 時系列データ分割レポート

生成日時: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}

## 分割概要

### 目的
データリーク防止のための時系列分割を実行しました。

### 分割方針
- **IRL学習**: 2019-2021年のデータでexpert trajectoriesから報酬関数を学習
- **RL訓練**: 2022年のデータで強化学習エージェントを訓練
- **テスト**: 2023年のデータで最終性能を評価

## データ統計

### 年別タスク数
"""
    
    for year, tasks in year_data.items():
        count = len(tasks)
        percentage = (count / total_tasks * 100) if total_tasks > 0 else 0
        report_content += f"- **{year}年**: {count:,}タスク ({percentage:.1f}%)\n"
    
    report_content += f"""
### 分割結果
- **IRL用データ**: {irl_count:,}タスク (2019-2021年)
- **RL訓練用データ**: {len(year_data['2022']):,}タスク (2022年)
- **テスト用データ**: {len(year_data['2023']):,}タスク (2023年)

## 生成ファイル

### データファイル
- `{new_paths['irl']}` - IRL学習用
- `{new_paths['training']}` - RL訓練用  
- `{new_paths['test']}` - テスト用

## 次のステップ

1. IRL学習の再実行 (2019-2021年データ)
2. 新しい設定でのRL訓練 (2022年データ)
3. 2023年データでの最終評価

## 実行コマンド

```bash
# 1. IRL学習の再実行（必要に応じて）
python scripts/create_expert_trajectories.py --data-path {new_paths['irl']}

# 2. RL訓練の実行
python scripts/train_simple_unified_rl.py

# 3. 2023年データでの評価
python scripts/evaluate_temporal_split.py --test-data {new_paths['test']}
```
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"   ✅ レポート生成完了")


def main():
    parser = argparse.ArgumentParser(description='時系列データ分割スクリプト')
    parser.add_argument('--input-file', default='data/backlog_with_2023.json',
                       help='2023年データを含む入力ファイル')
    parser.add_argument('--output-dir', default='data/temporal_split',
                       help='分割データの出力ディレクトリ')
    parser.add_argument('--backup-dir', default='backups/temporal_split',
                       help='既存設定のバックアップディレクトリ')
    parser.add_argument('--config-path', default='configs/unified_rl.yaml',
                       help='更新する設定ファイルのパス')
    parser.add_argument('--dry-run', action='store_true',
                       help='実際の変更を行わずに確認のみ')
    
    args = parser.parse_args()
    
    print("🚀 時系列データ分割スクリプト開始")
    print("=" * 60)
    
    if args.dry_run:
        print("⚠️ DRY RUN モード - 実際の変更は行いません")
        print()
    
    try:
        # 1. 入力ファイルの確認
        if not os.path.exists(args.input_file):
            print(f"❌ 入力ファイルが見つかりません: {args.input_file}")
            print("   2023年データを含むバックログファイルを準備してください。")
            return
        
        # 2. データ読み込みと分割
        data = load_backlog_data(args.input_file)
        year_data = split_data_by_year(data)
        
        # 3. 2023年データの確認
        if len(year_data['2023']) == 0:
            print("⚠️ 2023年のデータが見つかりません。")
            print("   2023年データを追加してから再実行してください。")
            return
        
        if args.dry_run:
            print("📊 分割結果プレビュー:")
            irl_count = len(year_data['2019']) + len(year_data['2020']) + len(year_data['2021'])
            print(f"   IRL用 (2019-2021): {irl_count:,}タスク")
            print(f"   RL訓練用 (2022): {len(year_data['2022']):,}タスク")
            print(f"   テスト用 (2023): {len(year_data['2023']):,}タスク")
            print("\n   実際に実行するには --dry-run フラグを外してください。")
            return
        
        # 4. 既存ファイルのバックアップ
        backup_existing_files(args.backup_dir)
        
        # 5. 分割データセットの作成
        new_paths = create_split_datasets(year_data, args.output_dir)
        
        # 6. 設定ファイルの更新
        update_config_file(new_paths, args.config_path)
        
        # 7. 移行レポートの作成
        create_migration_report(year_data, new_paths, args.output_dir)
        
        print("\n✅ 時系列データ分割完了！")
        print("=" * 60)
        print("📋 次のステップ:")
        print("   1. IRL学習の再実行 (2019-2021年データ)")
        print("   2. 新しい設定でのRL訓練 (2022年データ)")
        print("   3. 2023年データでの最終評価")
        print(f"\n📊 詳細は移行レポートを確認: {args.output_dir}/temporal_split_report.md")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
