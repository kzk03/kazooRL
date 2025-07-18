#!/usr/bin/env python3
"""
expert_trajectories.pklの詳細分析

開発者数、軌跡数、データ形式の確認
"""

import json
import pickle
from collections import defaultdict
from datetime import datetime


def analyze_expert_trajectories():
    """expert_trajectories.pklの詳細分析"""
    
    print("=" * 80)
    print("🔍 Expert Trajectories 詳細分析")
    print("=" * 80)
    
    trajectory_path = "data/expert_trajectories.pkl"
    
    try:
        with open(trajectory_path, "rb") as f:
            trajectories = pickle.load(f)
        
        print(f"✅ Successfully loaded: {trajectory_path}")
        print(f"📊 Trajectories type: {type(trajectories)}")
        print(f"📊 Number of trajectories: {len(trajectories)}")
        
    except FileNotFoundError:
        print(f"❌ File not found: {trajectory_path}")
        return
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    if not trajectories:
        print("⚠️ No trajectories found")
        return
    
    # 最初の軌跡を詳細分析
    first_trajectory = trajectories[0]
    print(f"\n📋 First trajectory analysis:")
    print(f"   Type: {type(first_trajectory)}")
    print(f"   Length: {len(first_trajectory)} steps")
    
    # 開発者の統計
    developers = set()
    tasks = set()
    step_count = 0
    
    print(f"\n🔍 Analyzing all steps...")
    
    for step_idx, step_data in enumerate(first_trajectory):
        step_count += 1
        
        if step_idx < 5:  # 最初の5ステップを詳細表示
            print(f"\n   Step {step_idx + 1}:")
            print(f"     Keys: {list(step_data.keys())}")
            
            if "action_details" in step_data:
                action_details = step_data["action_details"]
                developer_id = action_details.get("developer")
                task_id = action_details.get("task_id")
                timestamp = action_details.get("timestamp")
                
                print(f"     Developer: {developer_id}")
                print(f"     Task ID: {task_id}")
                print(f"     Timestamp: {timestamp}")
                
                if developer_id:
                    developers.add(developer_id)
                if task_id:
                    tasks.add(task_id)
        else:
            # 残りのステップは開発者とタスクIDだけ抽出
            if "action_details" in step_data:
                action_details = step_data["action_details"]
                developer_id = action_details.get("developer")
                task_id = action_details.get("task_id")
                
                if developer_id:
                    developers.add(developer_id)
                if task_id:
                    tasks.add(task_id)
    
    print(f"\n📊 Statistics:")
    print(f"   Total steps: {step_count:,}")
    print(f"   Unique developers: {len(developers)}")
    print(f"   Unique tasks: {len(tasks)}")
    
    # 開発者リスト表示
    developers_list = sorted(list(developers))
    print(f"\n👥 Developers in trajectories:")
    for i, dev in enumerate(developers_list):
        if i < 20:  # 最初の20人表示
            print(f"   {i+1:2d}. {dev}")
        elif i == 20:
            print(f"   ... and {len(developers_list) - 20} more")
            break
    
    # タスクリスト表示（一部）
    tasks_list = sorted(list(tasks))
    print(f"\n📋 Tasks in trajectories (first 10):")
    for i, task in enumerate(tasks_list[:10]):
        print(f"   {i+1:2d}. {task}")
    if len(tasks_list) > 10:
        print(f"   ... and {len(tasks_list) - 10} more")
    
    # 開発者ごとの活動統計
    dev_activity = defaultdict(int)
    task_activity = defaultdict(int)
    
    for step_data in first_trajectory:
        if "action_details" in step_data:
            action_details = step_data["action_details"]
            developer_id = action_details.get("developer")
            task_id = action_details.get("task_id")
            
            if developer_id:
                dev_activity[developer_id] += 1
            if task_id:
                task_activity[task_id] += 1
    
    print(f"\n📈 Developer Activity (Top 10):")
    dev_sorted = sorted(dev_activity.items(), key=lambda x: x[1], reverse=True)
    for i, (dev, count) in enumerate(dev_sorted[:10]):
        print(f"   {i+1:2d}. {dev}: {count} actions")
    
    print(f"\n📈 Task Activity (Top 10):")
    task_sorted = sorted(task_activity.items(), key=lambda x: x[1], reverse=True)
    for i, (task, count) in enumerate(task_sorted[:10]):
        print(f"   {i+1:2d}. {task}: {count} assignments")
    
    # 設定ファイルの開発者数と比較
    print(f"\n🔄 Configuration Comparison:")
    
    configs_to_check = [
        "configs/rl_debug.yaml",
        "configs/rl_experiment.yaml",
        "configs/base_training.yaml"
    ]
    
    for config_path in configs_to_check:
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            num_developers = config.get('num_developers', 'Not specified')
            print(f"   {config_path}: {num_developers}")
            
            if isinstance(num_developers, int):
                if num_developers == len(developers):
                    print(f"     ✅ Matches trajectory developers ({len(developers)})")
                else:
                    print(f"     ❌ Mismatch! Trajectory has {len(developers)} developers")
            
        except FileNotFoundError:
            print(f"   {config_path}: File not found")
        except Exception as e:
            print(f"   {config_path}: Error reading - {e}")
    
    print(f"\n💡 Recommendations:")
    trajectory_devs = len(developers)
    
    if trajectory_devs == 20:
        print(f"   ✅ 現在の軌跡は20人なので、rl_debug.yamlと一致")
        print(f"   💡 そのまま20人設定で実験を続行可能")
    elif trajectory_devs < 20:
        print(f"   ⚠️ 軌跡の開発者数({trajectory_devs})が20人より少ない")
        print(f"   💡 num_developers={trajectory_devs}に設定するか、軌跡を再生成")
    else:
        print(f"   💡 軌跡の開発者数({trajectory_devs})が20人より多い")
        print(f"   💡 最初の20人を使用するか、全員を使うよう設定変更")
    
    print(f"\n🚀 Next Actions:")
    if trajectory_devs != 20:
        print(f"   1. configs/rl_debug.yamlのnum_developersを{trajectory_devs}に変更")
        print(f"   2. または、20人用のexpert_trajectories.pklを再生成")
        print(f"   3. IRL重みも同じ開発者数で再訓練")
    else:
        print(f"   1. ✅ 現在の設定のまま実験続行可能")
        print(f"   2. IRL重みが20人分で訓練済みか確認")
    
    return {
        'num_developers': len(developers),
        'num_tasks': len(tasks),
        'num_steps': step_count,
        'developers_list': developers_list,
        'dev_activity': dict(dev_activity)
    }

if __name__ == "__main__":
    result = analyze_expert_trajectories()
