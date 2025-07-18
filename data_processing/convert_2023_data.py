#!/usr/bin/env python3
"""
2023年のGHArchiveデータからbacklog形式に変換
"""

import glob
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def process_github_event(event):
    """GitHubイベントからissue/PRデータを抽出"""
    if event['type'] not in ['IssuesEvent', 'PullRequestEvent']:
        return None
    
    payload = event['payload']
    
    # IssuesEventの場合
    if event['type'] == 'IssuesEvent':
        if 'issue' not in payload:
            return None
        
        issue = payload['issue']
        
        # 基本情報
        task_data = {
            'id': issue['id'],
            'number': issue['number'],
            'title': issue['title'],
            'body': issue.get('body', ''),
            'state': issue['state'],
            'created_at': issue['created_at'],
            'updated_at': issue['updated_at'],
            'closed_at': issue.get('closed_at'),
            'author': {
                'login': issue['user']['login'],
                'id': issue['user']['id']
            },
            'assignees': issue.get('assignees', []),
            'labels': [label['name'] for label in issue.get('labels', [])],
            'comments_count': issue.get('comments', 0),
            'repo': {
                'name': event['repo']['name'],
                'id': event['repo']['id']
            },
            'type': 'issue'
        }
        
        return task_data
    
    # PullRequestEventの場合
    elif event['type'] == 'PullRequestEvent':
        if 'pull_request' not in payload:
            return None
            
        pr = payload['pull_request']
        
        # 基本情報
        task_data = {
            'id': pr['id'],
            'number': pr['number'],
            'title': pr['title'],
            'body': pr.get('body', ''),
            'state': pr['state'],
            'created_at': pr['created_at'],
            'updated_at': pr['updated_at'],
            'closed_at': pr.get('closed_at'),
            'merged_at': pr.get('merged_at'),
            'author': {
                'login': pr['user']['login'],
                'id': pr['user']['id']
            },
            'assignees': pr.get('assignees', []),
            'labels': [label['name'] for label in pr.get('labels', [])],
            'comments_count': pr.get('comments', 0),
            'repo': {
                'name': event['repo']['name'],
                'id': event['repo']['id']
            },
            'type': 'pull_request'
        }
        
        return task_data
    
    return None


def main():
    print("🔄 2023年データ変換開始")
    print("=" * 60)
    
    # 2023年のJSONLファイルを取得
    data_dir = Path("/Users/kazuki-h/rl/kazoo/data/status/2023")
    jsonl_files = list(data_dir.glob("*.jsonl"))
    
    print(f"📂 対象ファイル: {len(jsonl_files)} 個")
    for file in sorted(jsonl_files):
        print(f"   {file.name}")
    
    # データ統合
    all_tasks = []
    issue_count = 0
    pr_count = 0
    other_count = 0
    
    for jsonl_file in sorted(jsonl_files):
        print(f"\n📖 処理中: {jsonl_file.name}")
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    event = json.loads(line.strip())
                    
                    task_data = process_github_event(event)
                    if task_data:
                        all_tasks.append(task_data)
                        
                        if task_data['type'] == 'issue':
                            issue_count += 1
                        elif task_data['type'] == 'pull_request':
                            pr_count += 1
                    else:
                        other_count += 1
                        
                except json.JSONDecodeError as e:
                    print(f"   ⚠️ JSON解析エラー (行 {line_num}): {e}")
                    continue
                except Exception as e:
                    print(f"   ⚠️ 処理エラー (行 {line_num}): {e}")
                    continue
        
        print(f"   処理済み: Issues={issue_count}, PRs={pr_count}, その他={other_count}")
    
    print(f"\n📊 抽出結果:")
    print(f"   Issues: {issue_count:,} 件")
    print(f"   Pull Requests: {pr_count:,} 件")
    print(f"   総計: {len(all_tasks):,} 件")
    
    if not all_tasks:
        print("❌ 抽出されたタスクがありません")
        return
    
    # 日付でソート
    all_tasks.sort(key=lambda x: x['created_at'])
    
    # 統計情報
    print(f"\n📅 期間情報:")
    print(f"   最古: {all_tasks[0]['created_at']}")
    print(f"   最新: {all_tasks[-1]['created_at']}")
    
    # 月別統計
    monthly_stats = defaultdict(int)
    for task in all_tasks:
        month = task['created_at'][:7]  # YYYY-MM
        monthly_stats[month] += 1
    
    print(f"\n📈 月別統計:")
    for month in sorted(monthly_stats.keys()):
        print(f"   {month}: {monthly_stats[month]:,} 件")
    
    # backlog_test_2023.json として保存
    output_path = Path("/Users/kazuki-h/rl/kazoo/data/backlog_test_2023.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_tasks, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 保存完了: {output_path}")
    print(f"   ファイルサイズ: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # 既存のbacklog.jsonと統合
    main_backlog_path = Path("/Users/kazuki-h/rl/kazoo/data/backlog.json")
    if main_backlog_path.exists():
        print(f"\n🔗 既存データとの統合...")
        
        with open(main_backlog_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        print(f"   既存データ: {len(existing_data):,} 件")
        
        # 重複チェック（IDベース）
        existing_ids = set(item.get('id') for item in existing_data if item.get('id'))
        new_tasks = [task for task in all_tasks if task['id'] not in existing_ids]
        
        print(f"   新規データ: {len(new_tasks):,} 件")
        print(f"   重複除外: {len(all_tasks) - len(new_tasks):,} 件")
        
        # 統合
        combined_data = existing_data + new_tasks
        combined_data.sort(key=lambda x: x.get('created_at', ''))
        
        # バックアップ
        backup_path = main_backlog_path.with_suffix('.bak')
        main_backlog_path.rename(backup_path)
        print(f"   バックアップ: {backup_path}")
        
        # 新しいbacklog.json保存
        with open(main_backlog_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 統合完了: {len(combined_data):,} 件")
        
        # 年次分布の再表示
        years = defaultdict(int)
        for item in combined_data:
            year = item.get('created_at', '')[:4]
            if year:
                years[year] += 1
        
        print(f"\n📊 統合後の年次分布:")
        for year in sorted(years.keys()):
            print(f"   {year}年: {years[year]:,} 件")
    
    print(f"\n⚠️ 次のステップ:")
    print(f"   1. python results/scripts/create_temporal_split.py を実行")
    print(f"   2. 2019-2021: IRL, 2022: RL訓練, 2023: テスト用に分割")
    print(f"   3. evaluation/behavioral_comparison.py でテスト実行")


if __name__ == "__main__":
    main()
