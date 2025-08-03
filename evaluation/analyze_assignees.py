#!/usr/bin/env python3
"""
実際の担当者データの分析
"""

import json
import sys
from collections import Counter
from pathlib import Path


def analyze_assignees(test_data_path: str):
    """担当者データの分析"""
    print(f"📊 担当者データ分析: {test_data_path}")
    
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    all_assignees = []
    tasks_with_assignees = 0
    
    print("\n🔍 担当者情報の詳細分析:")
    
    for i, task in enumerate(test_data[:10]):  # 最初の10個を詳細表示
        assignees = task.get("assignees", [])
        print(f"\nタスク {i+1}:")
        print(f"  ID: {task.get('id')}")
        print(f"  Title: {task.get('title', '')[:50]}...")
        print(f"  Assignees: {assignees}")
        
        if assignees:
            tasks_with_assignees += 1
            for assignee in assignees:
                if isinstance(assignee, dict):
                    login = assignee.get("login", "unknown")
                    all_assignees.append(login)
                    print(f"    - {login}")
                else:
                    all_assignees.append(str(assignee))
                    print(f"    - {assignee}")
    
    # 全体統計
    total_with_assignees = sum(1 for task in test_data if task.get("assignees"))
    assignee_counter = Counter(all_assignees)
    
    print(f"\n📈 全体統計:")
    print(f"  総タスク数: {len(test_data):,}")
    print(f"  担当者ありタスク数: {total_with_assignees:,}")
    print(f"  担当者あり率: {total_with_assignees/len(test_data)*100:.1f}%")
    print(f"  ユニーク担当者数: {len(assignee_counter)}")
    
    print(f"\n👥 上位担当者:")
    for assignee, count in assignee_counter.most_common(10):
        print(f"  {assignee}: {count}タスク")
    
    # 訓練されたエージェント名との比較
    import os
    model_dir = "models/improved_rl/final_models"
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        trained_agents = [f.replace('agent_', '').replace('.pth', '') for f in model_files[:10]]
        
        print(f"\n🤖 訓練されたエージェント例:")
        for agent in trained_agents:
            print(f"  {agent}")
        
        # 一致する名前があるかチェック
        actual_assignees = set(all_assignees)
        trained_set = set(trained_agents)
        matches = actual_assignees.intersection(trained_set)
        
        print(f"\n🎯 名前の一致:")
        print(f"  一致する名前数: {len(matches)}")
        if matches:
            print(f"  一致する名前: {list(matches)}")
        else:
            print("  一致する名前なし - 代替評価方法が必要")

if __name__ == "__main__":
    analyze_assignees("data/backlog_test_2023.json")