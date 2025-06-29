#!/usr/bin/env python3
"""
Expert軌跡生成スクリプト（Bot除外版）

オリジナルのcreate_expert_trajectories.pyを基に、以下の機能を追加：
1. Bot開発者（[bot]を含む名前）を除外
2. Bot開発者が関与したイベント・タスクを除外
3. 人間の開発者のみの軌跡を生成

使用例:
    python tools/analysis/create_expert_trajectories_bot_excluded.py
"""

import glob
import json
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import yaml


class Action:
    """アクション定義"""
    MERGE_PULL_REQUEST = "MERGE_PULL_REQUEST"
    CLOSE_ISSUE = "CLOSE_ISSUE"


def is_bot_developer(developer_name: str) -> bool:
    """開発者名がボットかどうかを判定"""
    if not developer_name:
        return True
    return '[bot]' in developer_name.lower()


def load_human_developers(dev_profiles_path: str) -> Set[str]:
    """人間の開発者リストを読み込み（ボット除外）"""
    try:
        with open(dev_profiles_path, "r", encoding="utf-8") as f:
            dev_profiles = yaml.safe_load(f)
        
        human_developers = set()
        bot_developers = set()
        
        for dev_name in dev_profiles.keys():
            if is_bot_developer(dev_name):
                bot_developers.add(dev_name)
            else:
                human_developers.add(dev_name)
        
        print(f"Loaded developer profiles:")
        print(f"  Human developers: {len(human_developers)}")
        print(f"  Bot developers: {len(bot_developers)}")
        if bot_developers:
            print(f"  Bot developers excluded: {list(bot_developers)[:5]}{'...' if len(bot_developers) > 5 else ''}")
        
        return human_developers
        
    except Exception as e:
        print(f"Error loading developer profiles from {dev_profiles_path}: {e}")
        print("Warning: Proceeding without developer profile filtering")
        return set()


def filter_bot_tasks(all_tasks: List[Dict], human_developers: Set[str]) -> List[Dict]:
    """ボット開発者が関与したタスクを除外"""
    human_tasks = []
    bot_task_count = 0
    
    for task in all_tasks:
        # assigned_toでボット判定
        assigned_to = task.get('assigned_to', '')
        
        # ボット開発者に割り当てられたタスクを除外
        if assigned_to and is_bot_developer(assigned_to):
            bot_task_count += 1
            continue
        
        # 開発者プロファイルがある場合、リストにない開発者のタスクも除外
        if human_developers and assigned_to and assigned_to not in human_developers:
            bot_task_count += 1
            continue
        
        human_tasks.append(task)
    
    print(f"Task filtering:")
    print(f"  Original tasks: {len(all_tasks)}")
    print(f"  Bot tasks filtered out: {bot_task_count}")
    print(f"  Human tasks remaining: {len(human_tasks)}")
    
    return human_tasks


def map_event_to_action(event: Dict, human_developers: Set[str]) -> Tuple[Optional[str], Optional[Dict]]:
    """
    GitHubイベントをActionにマッピング（ボット除外版）
    
    Args:
        event: GitHubイベント
        human_developers: 人間の開発者セット
    
    Returns:
        (action_enum, action_details) or (None, None)
    """
    event_type = event.get("type")

    # 1. PRのマージイベントをアクションに変換
    if (
        event_type == "PullRequestEvent"
        and event.get("payload", {}).get("action") == "closed"
        and event.get("payload", {}).get("pull_request", {}).get("merged")
    ):
        pr = event["payload"]["pull_request"]
        developer = pr.get("user", {}).get("login")
        
        # ボット開発者を除外
        if is_bot_developer(developer):
            return None, None
        
        # 開発者プロファイルがある場合、リストにない開発者も除外
        if human_developers and developer not in human_developers:
            return None, None

        return Action.MERGE_PULL_REQUEST, {
            "task_id": pr.get("id"),
            "developer": developer,
            "timestamp": pr.get("merged_at"),
        }

    # 2. Issueのクローズイベントをアクションに変換
    elif (
        event_type == "IssuesEvent"
        and event.get("payload", {}).get("action") == "closed"
    ):
        issue = event["payload"]["issue"]
        
        # PRに紐づくIssueは除外
        if "pull_request" in issue:
            return None, None

        developer = event.get("actor", {}).get("login")
        
        # ボット開発者を除外
        if is_bot_developer(developer):
            return None, None
        
        # 開発者プロファイルがある場合、リストにない開発者も除外
        if human_developers and developer not in human_developers:
            return None, None

        return Action.CLOSE_ISSUE, {
            "task_id": issue.get("id"),
            "developer": developer,
            "timestamp": issue.get("closed_at"),
        }

    return None, None


def main(data_dir: str, backlog_path: str, dev_profiles_path: str, output_path: str):
    """
    Bot除外版Expert軌跡生成のメイン処理
    
    Args:
        data_dir: イベントデータディレクトリ
        backlog_path: バックログファイルパス
        dev_profiles_path: 開発者プロファイルファイルパス
        output_path: 出力ファイルパス
    """
    print("🚀 Creating Bot-Excluded Expert Trajectories")
    print("=" * 60)
    
    # 人間の開発者リストを読み込み
    human_developers = load_human_developers(dev_profiles_path)
    
    # 複数の.jsonlファイルを読み込む
    print(f"\nReading event data from directory: {data_dir}")
    jsonl_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    if not jsonl_files:
        print(f"Error: No .jsonl files found in {data_dir}")
        return

    all_events = []
    bot_events_filtered = 0
    
    for file_path in sorted(jsonl_files):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    event = json.loads(line)
                    
                    # イベントレベルでのボット除外チェック
                    actor = event.get("actor", {}).get("login", "")
                    if is_bot_developer(actor):
                        bot_events_filtered += 1
                        continue
                    
                    # 開発者プロファイルがある場合の追加チェック
                    if human_developers and actor and actor not in human_developers:
                        bot_events_filtered += 1
                        continue
                    
                    all_events.append(event)

    # イベントをタイムスタンプでソートする
    all_events.sort(key=lambda x: x.get("created_at") or "")
    print(f"Event filtering:")
    print(f"  Bot events filtered out: {bot_events_filtered}")
    print(f"  Human events remaining: {len(all_events)}")

    # バックログデータの読み込みと前処理
    try:
        with open(backlog_path, "r", encoding="utf-8") as f:
            all_tasks_list = json.load(f)
        
        # ボットタスクを除外
        human_tasks_list = filter_bot_tasks(all_tasks_list, human_developers)
        all_tasks_db = {task["id"]: task for task in human_tasks_list}
        
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading backlog file '{backlog_path}': {e}")
        return

    # メインループ：軌跡生成
    print(f"\nGenerating expert trajectories...")
    trajectory = []
    human_actions_count = 0
    bot_actions_filtered = 0
    
    for event in all_events:
        action_enum, action_details = map_event_to_action(event, human_developers)

        # マッピングされたアクションがなければスキップ
        if not action_enum:
            continue

        event_timestamp = action_details.get("timestamp")
        developer = action_details.get("developer")
        task_id = action_details.get("task_id")

        if not all([event_timestamp, developer, task_id]):
            continue

        # タスクがボット除外済みリストにあるかチェック
        if task_id not in all_tasks_db:
            bot_actions_filtered += 1
            continue

        # イベント発生直前の「状態」を定義
        open_tasks_at_event = {
            task_id
            for task_id, task_data in all_tasks_db.items()
            if (
                task_data.get("created_at")
                and task_data.get("created_at") <= event_timestamp
            )
            and (
                not task_data.get("closed_at")
                or task_data.get("closed_at") > event_timestamp
            )
        }

        current_state = {"open_task_ids": list(open_tasks_at_event)}

        # (状態, 行動)のペアを軌跡に追加
        trajectory.append(
            {
                "state": current_state,
                "action": action_enum,
                "action_details": action_details,
            }
        )
        human_actions_count += 1

    # 軌跡の保存
    trajectories = [trajectory] if trajectory else []

    with open(output_path, "wb") as f:
        pickle.dump(trajectories, f)
    
    # 結果の報告
    print(f"\n📊 Expert Trajectories Generation Summary:")
    print(f"  Human actions included: {human_actions_count}")
    print(f"  Bot actions filtered out: {bot_actions_filtered}")
    print(f"  Total trajectories: {len(trajectories)}")
    print(f"  Total steps in trajectory: {len(trajectory)}")
    print(f"  Output saved to: {output_path}")
    
    # 軌跡の内容確認
    if trajectory:
        print(f"\n📋 Sample trajectory steps:")
        for i, step in enumerate(trajectory[:5]):
            action_details = step['action_details']
            developer = action_details.get('developer', 'Unknown')
            action = step['action']
            task_id = action_details.get('task_id', 'Unknown')
            print(f"  {i+1}. {developer} -> {action} (Task: {task_id})")
        
        if len(trajectory) > 5:
            print(f"  ... and {len(trajectory) - 5} more steps")
    
    print("\n✅ Bot-excluded expert trajectories created successfully!")


if __name__ == "__main__":
    # デフォルト設定
    INPUT_DATA_DIR = "./data"  # 複数年分のデータを含むディレクトリ
    BACKLOG_FILE_PATH = "data/backlog.json"
    DEV_PROFILES_PATH = "configs/dev_profiles.yaml"
    OUTPUT_TRAJECTORY_PATH = "data/expert_trajectories_bot_excluded.pkl"
    
    print(f"Input configuration:")
    print(f"  Data directory: {INPUT_DATA_DIR}")
    print(f"  Backlog file: {BACKLOG_FILE_PATH}")
    print(f"  Developer profiles: {DEV_PROFILES_PATH}")
    print(f"  Output file: {OUTPUT_TRAJECTORY_PATH}")
    print()
    
    main(INPUT_DATA_DIR, BACKLOG_FILE_PATH, DEV_PROFILES_PATH, OUTPUT_TRAJECTORY_PATH)
