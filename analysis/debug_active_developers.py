#!/usr/bin/env python3
"""
アクティブ開発者フィルタリングのデバッグスクリプト
"""

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import yaml


def parse_datetime(date_str):
    """日付文字列をdatetimeオブジェクトに変換"""
    try:
        if date_str.endswith("Z"):
            date_str = date_str[:-1] + "+00:00"
        return datetime.fromisoformat(date_str)
    except:
        return None

def debug_active_developers():
    """アクティブ開発者フィルタリングをデバッグ"""
    
    print("🔍 アクティブ開発者フィルタリング デバッグ開始")
    
    # データ読み込み
    with open("data/backlog_test_2023.json", "r", encoding="utf-8") as f:
        test_tasks = json.load(f)
    
    with open("configs/dev_profiles_test_2023.yaml", "r", encoding="utf-8") as f:
        dev_profiles = yaml.safe_load(f)
    
    print(f"📊 テストタスク数: {len(test_tasks)}")
    print(f"👥 プロファイル開発者数: {len(dev_profiles)}")
    
    # 担当者情報があるタスクを抽出
    tasks_with_assignees = []
    for task in test_tasks:
        if task.get("assignees") and len(task["assignees"]) > 0:
            assignees = [a.get("login") for a in task["assignees"] if a.get("login")]
            if any(assignee in dev_profiles for assignee in assignees):
                tasks_with_assignees.append(task)
    
    print(f"📋 担当者情報があるタスク: {len(tasks_with_assignees)}")
    
    # 最初の数タスクでアクティブ開発者フィルタリングをテスト
    print(f"\n🧪 最初の5タスクでアクティブ開発者フィルタリングをテスト:")
    
    for i, task in enumerate(tasks_with_assignees[:5]):
        print(f"\n--- タスク {i+1} ---")
        print(f"ID: {task.get('id')}")
        print(f"タイトル: {task.get('title', '')[:50]}")
        print(f"更新日: {task.get('updated_at', '')}")
        
        # 実際の担当者
        actual_assignees = [a.get("login") for a in task["assignees"] if a.get("login")]
        print(f"実際の担当者: {actual_assignees}")
        
        # タスクの日付を取得
        task_date = parse_datetime(task.get("updated_at", task.get("created_at", "")))
        if not task_date:
            print("❌ タスク日付が取得できません")
            continue
        
        print(f"タスク日付: {task_date}")
        
        # 活動期間を設定（30日、90日、180日で比較）
        for activity_days in [30, 90, 180]:
            activity_start = task_date - timedelta(days=activity_days)
            activity_end = task_date + timedelta(days=7)
            
            print(f"\n  📅 活動期間 {activity_days}日:")
            print(f"    期間: {activity_start.date()} ～ {activity_end.date()}")
            
            # 活動期間内でアクティブな開発者を抽出
            active_developers = set()
            
            for other_task in test_tasks:
                other_task_date = parse_datetime(other_task.get("updated_at", other_task.get("created_at", "")))
                if not other_task_date:
                    continue
                    
                # 活動期間内のタスクかチェック
                if activity_start <= other_task_date <= activity_end:
                    # タスクの担当者を追加
                    if other_task.get("assignees"):
                        for assignee in other_task["assignees"]:
                            if assignee.get("login") and assignee["login"] in dev_profiles:
                                active_developers.add(assignee["login"])
                    
                    # タスクの作成者も追加
                    author = other_task.get("user", other_task.get("author", {}))
                    if author and author.get("login") and author["login"] in dev_profiles:
                        active_developers.add(author["login"])
            
            print(f"    アクティブ開発者数: {len(active_developers)}")
            print(f"    アクティブ開発者: {sorted(list(active_developers))}")
            
            # 実際の担当者がアクティブ開発者に含まれているかチェック
            actual_in_active = [assignee for assignee in actual_assignees if assignee in active_developers]
            print(f"    実際の担当者がアクティブに含まれる: {actual_in_active}")
    
    # 全体的な活動パターンの分析
    print(f"\n📊 全体的な活動パターン分析:")
    
    # 月別の活動統計
    monthly_stats = defaultdict(lambda: {"tasks": 0, "developers": set(), "assignees": set()})
    
    for task in test_tasks:
        task_date = parse_datetime(task.get("updated_at", task.get("created_at", "")))
        if not task_date:
            continue
        
        month_key = task_date.strftime("%Y-%m")
        monthly_stats[month_key]["tasks"] += 1
        
        # 担当者を記録
        if task.get("assignees"):
            for assignee in task["assignees"]:
                if assignee.get("login") and assignee["login"] in dev_profiles:
                    monthly_stats[month_key]["assignees"].add(assignee["login"])
        
        # 作成者を記録
        author = task.get("user", task.get("author", {}))
        if author and author.get("login") and author["login"] in dev_profiles:
            monthly_stats[month_key]["developers"].add(author["login"])
    
    print(f"\n📈 月別活動統計（詳細）:")
    for month in sorted(monthly_stats.keys())[-12:]:  # 直近12ヶ月
        stats = monthly_stats[month]
        total_devs = len(stats["developers"] | stats["assignees"])
        print(f"  {month}: {stats['tasks']:3d}タスク, {len(stats['assignees']):2d}担当者, {len(stats['developers']):2d}作成者, {total_devs:2d}総開発者")
        if stats["assignees"]:
            print(f"    担当者: {sorted(list(stats['assignees']))}")
    
    # 推薦システムの問題を特定
    print(f"\n🔍 推薦システムの問題分析:")
    
    # 実際の担当者の活動パターン
    actual_assignees_all = set()
    for task in tasks_with_assignees:
        for assignee in task["assignees"]:
            if assignee.get("login"):
                actual_assignees_all.add(assignee["login"])
    
    print(f"実際の担当者（全体）: {sorted(list(actual_assignees_all))}")
    
    # 各担当者の最初と最後の活動日
    assignee_activity = defaultdict(list)
    for task in test_tasks:
        task_date = parse_datetime(task.get("updated_at", task.get("created_at", "")))
        if not task_date:
            continue
        
        if task.get("assignees"):
            for assignee in task["assignees"]:
                if assignee.get("login") and assignee["login"] in actual_assignees_all:
                    assignee_activity[assignee["login"]].append(task_date)
    
    print(f"\n👤 担当者別活動期間:")
    for assignee in sorted(actual_assignees_all):
        if assignee in assignee_activity:
            dates = sorted(assignee_activity[assignee])
            print(f"  {assignee:15s}: {dates[0].date()} ～ {dates[-1].date()} ({len(dates)}タスク)")
        else:
            print(f"  {assignee:15s}: 活動記録なし")

if __name__ == "__main__":
    debug_active_developers()