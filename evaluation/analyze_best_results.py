#!/usr/bin/env python3
"""
最優秀手法の詳細分析
"""

import json
import os
from collections import Counter, defaultdict
from typing import Dict, List

from advanced_ensemble_system import AdvancedEnsembleSystem


def analyze_meta_ensemble_details():
    """メタアンサンブルの詳細分析"""
    print("🔍 メタアンサンブル手法の詳細分析")
    print("=" * 50)
    
    # システム初期化
    system = AdvancedEnsembleSystem(
        model_dir="models/improved_rl/final_models",
        test_data_path="data/backlog_test_2023.json",
    )
    
    # 詳細分析用データ
    available_agents = set(system.models.keys())
    eval_tasks = []
    eval_ground_truth = []
    
    for task, author in zip(system.tasks[:200], system.ground_truth[:200]):
        if author in available_agents and len(eval_tasks) < 100:
            eval_tasks.append(task)
            eval_ground_truth.append(author)
    
    # 詳細分析
    correct_predictions = []
    incorrect_predictions = []
    author_performance = defaultdict(list)
    task_type_performance = defaultdict(list)
    
    for task, actual_author in zip(eval_tasks, eval_ground_truth):
        task_features = system._extract_task_features(task)
        recommendations = system.meta_ensemble_recommendation(task_features, task, 1)
        
        predicted_author = recommendations[0][0] if recommendations else None
        predicted_score = recommendations[0][1] if recommendations else 0.0
        
        is_correct = (predicted_author == actual_author)
        
        # タスクタイプ分類
        title_lower = (task.get("title", "") or "").lower()
        body_lower = (task.get("body", "") or "").lower()
        full_text = f"{title_lower} {body_lower}"
        
        if any(kw in full_text for kw in ["bug", "fix", "error", "issue"]):
            task_type = "bug"
        elif any(kw in full_text for kw in ["feature", "enhancement", "new"]):
            task_type = "feature"
        elif any(kw in full_text for kw in ["doc", "readme", "guide"]):
            task_type = "doc"
        else:
            task_type = "other"
        
        result = {
            "task_title": task.get("title", "")[:60] + "...",
            "task_type": task_type,
            "actual_author": actual_author,
            "predicted_author": predicted_author,
            "predicted_score": predicted_score,
            "is_correct": is_correct,
            "contribution": system.author_contributions.get(actual_author, 0),
        }
        
        if is_correct:
            correct_predictions.append(result)
        else:
            incorrect_predictions.append(result)
        
        author_performance[actual_author].append(is_correct)
        task_type_performance[task_type].append(is_correct)
    
    # 結果表示
    print(f"## 📊 分析結果")
    print(f"   総評価数: {len(eval_tasks)}")
    print(f"   正解数: {len(correct_predictions)}")
    print(f"   精度: {len(correct_predictions)/len(eval_tasks)*100:.1f}%")
    
    # タスクタイプ別性能
    print(f"\n### 📈 タスクタイプ別性能")
    for task_type, results in task_type_performance.items():
        accuracy = sum(results) / len(results) * 100
        print(f"   {task_type}: {accuracy:.1f}% ({sum(results)}/{len(results)})")
    
    # 最も成功した開発者
    print(f"\n### 🏆 高精度で予測された開発者 (Top-10)")
    author_accuracy = {}
    for author, results in author_performance.items():
        if len(results) >= 2:  # 2回以上出現
            author_accuracy[author] = sum(results) / len(results)
    
    sorted_authors = sorted(author_accuracy.items(), key=lambda x: x[1], reverse=True)
    for i, (author, accuracy) in enumerate(sorted_authors[:10]):
        contribution = system.author_contributions.get(author, 0)
        total_attempts = len(author_performance[author])
        print(f"   {i+1}. {author}: {accuracy*100:.1f}% ({int(accuracy*total_attempts)}/{total_attempts}) [貢献{contribution}]")
    
    # 成功事例分析
    print(f"\n### ✅ 成功事例 (Top-5)")
    correct_sorted = sorted(correct_predictions, key=lambda x: x["predicted_score"], reverse=True)
    for i, case in enumerate(correct_sorted[:5]):
        print(f"   {i+1}. タスク: {case['task_title']}")
        print(f"      タイプ: {case['task_type']}, 著者: {case['actual_author']}")
        print(f"      スコア: {case['predicted_score']:.3f}, 貢献: {case['contribution']}")
        print()
    
    # 失敗事例分析
    print(f"\n### ❌ 失敗事例 (高貢献者)")
    high_contrib_errors = [p for p in incorrect_predictions if p["contribution"] >= 20]
    high_contrib_errors.sort(key=lambda x: x["contribution"], reverse=True)
    
    for i, case in enumerate(high_contrib_errors[:3]):
        print(f"   {i+1}. タスク: {case['task_title']}")
        print(f"      実際: {case['actual_author']} (貢献{case['contribution']})")
        print(f"      予測: {case['predicted_author']} (スコア{case['predicted_score']:.3f})")
        print()


if __name__ == "__main__":
    analyze_meta_ensemble_details()