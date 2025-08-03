#!/usr/bin/env python3
"""
実際の作成者名と訓練されたエージェント名の重複をチェック
"""

import json
import os
from collections import Counter


def check_name_overlap():
    """名前の重複をチェック"""
    print("🔍 名前の重複チェック開始")
    
    # 1. テストデータから実際の作成者名を取得
    with open("data/backlog_test_2023.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    actual_authors = []
    for task in test_data:
        author = task.get("author", {})
        if author and isinstance(author, dict):
            author_login = author.get("login", "")
            if author_login:
                actual_authors.append(author_login)
    
    author_counter = Counter(actual_authors)
    print(f"📊 実際の作成者: {len(author_counter)}人")
    
    # 2. 訓練されたエージェント名を取得
    model_dir = "models/improved_rl/final_models"
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    trained_agents = [f.replace('agent_', '').replace('.pth', '') for f in model_files]
    
    print(f"🤖 訓練されたエージェント: {len(trained_agents)}人")
    
    # 3. 重複をチェック
    actual_set = set(actual_authors)
    trained_set = set(trained_agents)
    
    overlap = actual_set.intersection(trained_set)
    print(f"🎯 重複する名前: {len(overlap)}人")
    
    if overlap:
        print("✅ 重複する名前:")
        overlap_counter = {name: author_counter[name] for name in overlap}
        for name, count in sorted(overlap_counter.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {count}タスク")
        
        # 重複する名前でのタスク数
        overlap_tasks = sum(author_counter[name] for name in overlap)
        print(f"\n📈 重複名前でのタスク数: {overlap_tasks} / {len(actual_authors)} ({overlap_tasks/len(actual_authors)*100:.1f}%)")
    else:
        print("❌ 重複する名前なし")
        
        # 上位作成者と類似する訓練エージェント名を探す
        print("\n🔍 類似名前の検索:")
        top_authors = [name for name, _ in author_counter.most_common(20)]
        
        for author in top_authors:
            similar_agents = [agent for agent in trained_agents if author.lower() in agent.lower() or agent.lower() in author.lower()]
            if similar_agents:
                print(f"  {author} → 類似: {similar_agents[:3]}")
    
    # 4. 統計サマリー
    print(f"\n📊 統計サマリー:")
    print(f"  実際の作成者数: {len(actual_set):,}")
    print(f"  訓練エージェント数: {len(trained_set):,}")
    print(f"  重複数: {len(overlap):,}")
    print(f"  重複率: {len(overlap)/min(len(actual_set), len(trained_set))*100:.1f}%")
    
    return overlap, author_counter, trained_agents

if __name__ == "__main__":
    check_name_overlap()