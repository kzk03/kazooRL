#!/usr/bin/env python3
"""
時系列データ分割スクリプト
2023年データ追加後の理想的なデータ分割を実行

データリークを防ぐための時系列分割:
- IRL学習: 2019-2021年 (expert trajectories用)
- RL訓練: 2022年 (強化学習訓練用)
- テスト: 2023年 (最終評価用)
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import click
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
        'data/backlog_training.json',
        'data/backlog_test_2022.json'
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
    config['env']['expert_trajectories_path'] = 'data/expert_trajectories_2019_2021.pkl'  # 必要に応じて更新
    
    # 評価用設定を追加
    if 'evaluation' not in config:
        config['evaluation'] = {}
    config['evaluation']['test_data_path'] = new_paths['test']
    
    # コメントを追加
    config['# データ分割について'] = {
        'IRL学習期間': '2019-2021年',
        'RL訓練期間': '2022年',
        'テスト期間': '2023年',
        '分割理由': 'データリーク防止のための時系列分割'
    }
    
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
    
    total_tasks = sum(len(tasks) for tasks in year_data.values())
    
    for year, tasks in year_data.items():
        count = len(tasks)
        percentage = (count / total_tasks * 100) if total_tasks > 0 else 0
        report_content += f"- **{year}年**: {count:,}タスク ({percentage:.1f}%)\n"
    
    report_content += f"""
### 分割結果
- **IRL用データ**: {len(year_data['2019']) + len(year_data['2020']) + len(year_data['2021']):,}タスク (2019-2021年)
- **RL訓練用データ**: {len(year_data['2022']):,}タスク (2022年)
- **テスト用データ**: {len(year_data['2023']):,}タスク (2023年)

## 生成ファイル

### データファイル
- `{new_paths['irl']}` - IRL学習用
- `{new_paths['training']}` - RL訓練用  
- `{new_paths['test']}` - テスト用

### 設定ファイル
- `configs/unified_rl.yaml` - 更新済み

## 利点

1. **データリーク防止**: 訓練とテストで完全に異なる期間のデータを使用
2. **時系列順序**: 過去→現在→未来の順でデータを使用
3. **現実的評価**: 未来のデータでの性能評価により実用性を検証

## 次のステップ

1. IRL学習の再実行 (2019-2021年データ)
2. 新しい設定でのRL訓練 (2022年データ)
3. 2023年データでの最終評価

## 注意事項

- 既存の訓練済みモデルは2019-2021年データで学習されているため、新しい分割には適用できません
- IRLの再学習が必要です
- 評価結果は以前のものと直接比較できません
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"   ✅ レポート生成完了")


@click.command()
@click.option('--input-file', default='data/backlog_with_2023.json', 
              help='2023年データを含む入力ファイル')
@click.option('--output-dir', default='data/temporal_split', 
              help='分割データの出力ディレクトリ')
@click.option('--backup-dir', default='backups/temporal_split', 
              help='既存設定のバックアップディレクトリ')
@click.option('--config-path', default='configs/unified_rl.yaml', 
              help='更新する設定ファイルのパス')
@click.option('--dry-run', is_flag=True, 
              help='実際の変更を行わずに確認のみ')
def main(input_file: str, output_dir: str, backup_dir: str, config_path: str, dry_run: bool):
    """
    時系列データ分割スクリプト
    
    2023年データを含むバックログファイルを時系列で分割し、
    データリークを防ぐ理想的な構成を作成します。
    """
    
    print("🚀 時系列データ分割スクリプト開始")
    print("=" * 60)
    
    if dry_run:
        print("⚠️ DRY RUN モード - 実際の変更は行いません")
        print()
    
    try:
        # 1. 入力ファイルの確認
        if not os.path.exists(input_file):
            print(f"❌ 入力ファイルが見つかりません: {input_file}")
            print("   2023年データを含むバックログファイルを準備してください。")
            return
        
        # 2. データ読み込みと分割
        data = load_backlog_data(input_file)
        year_data = split_data_by_year(data)
        
        # 3. 2023年データの確認
        if len(year_data['2023']) == 0:
            print("⚠️ 2023年のデータが見つかりません。")
            print("   2023年データを追加してから再実行してください。")
            return
        
        if dry_run:
            print("📊 分割結果プレビュー:")
            irl_count = len(year_data['2019']) + len(year_data['2020']) + len(year_data['2021'])
            print(f"   IRL用 (2019-2021): {irl_count:,}タスク")
            print(f"   RL訓練用 (2022): {len(year_data['2022']):,}タスク")
            print(f"   テスト用 (2023): {len(year_data['2023']):,}タスク")
            print("\n   実際に実行するには --dry-run フラグを外してください。")
            return
        
        # 4. 既存ファイルのバックアップ
        backup_existing_files(backup_dir)
        
        # 5. 分割データセットの作成
        new_paths = create_split_datasets(year_data, output_dir)
        
        # 6. 設定ファイルの更新
        update_config_file(new_paths, config_path)
        
        # 7. 移行レポートの作成
        create_migration_report(year_data, new_paths, output_dir)
        
        print("\n✅ 時系列データ分割完了！")
        print("=" * 60)
        print("📋 次のステップ:")
        print("   1. IRL学習の再実行 (2019-2021年データ)")
        print("   2. 新しい設定でのRL訓練 (2022年データ)")
        print("   3. 2023年データでの最終評価")
        print(f"\n📊 詳細は移行レポートを確認: {output_dir}/temporal_split_report.md")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
