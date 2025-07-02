import glob
import json
import os
import pickle
from collections import defaultdict

# kazoo.consts.actions がENUM形式などで定義されていることを想定
# from kazoo.consts.actions import Action


# Actionの定義（仮）
class Action:
    MERGE_PULL_REQUEST = "MERGE_PULL_REQUEST"
    CLOSE_ISSUE = "CLOSE_ISSUE"
    # 他のアクションもここに追加可能
    # ASSIGN_TASK = "ASSIGN_TASK"
    # SUBMIT_PULL_REQUEST = "SUBMIT_PULL_REQUEST"
    # APPROVE_PULL_REQUEST = "APPROVE_PULL_REQUEST"


def map_event_to_action(event):
    """
    GitHubイベント（.jsonlの各行）をシミュレータのActionにマッピングする。
    PRのマージとIssueのクローズを「お手本」のアクションとして定義する。
    """
    event_type = event.get("type")

    # 1. PRのマージイベントをアクションに変換
    if (
        event_type == "PullRequestEvent"
        and event.get("payload", {}).get("action") == "closed"
        and event.get("payload", {}).get("pull_request", {}).get("merged")
    ):

        pr = event["payload"]["pull_request"]
        return Action.MERGE_PULL_REQUEST, {
            "task_id": pr.get("id"),
            "developer": pr.get("user", {}).get("login"),
            "timestamp": pr.get("merged_at"),  # アクションが発生した正確な時刻
        }

    # 2. Issueのクローズイベントをアクションに変換
    elif (
        event_type == "IssuesEvent"
        and event.get("payload", {}).get("action") == "closed"
    ):

        issue = event["payload"]["issue"]
        # PRに紐づくIssueは除外（重複を避けるため）
        if "pull_request" in issue:
            return None, None

        return Action.CLOSE_ISSUE, {
            "task_id": issue.get("id"),
            "developer": event.get("actor", {}).get("login"),  # Issueを閉じた人が実行者
            "timestamp": issue.get("closed_at"),
        }

    return None, None


def main(data_dir, backlog_path, output_path):
    # ▼▼▼【ここからが修正箇所】▼▼▼
    # 1. 複数の.jsonlファイルを読み込む（data/直下とstatus/20**以下の両方）
    print(f"Reading event data from directory: {data_dir}")
    
    jsonl_files = []
    # data_dir直下のjsonlファイル
    direct_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    jsonl_files.extend(direct_files)
    
    # data_dir/status/20** 以下のjsonlファイル
    status_pattern = os.path.join(data_dir, "status", "20*", "*.jsonl")
    status_files = glob.glob(status_pattern)
    jsonl_files.extend(status_files)
    
    if not jsonl_files:
        print(f"Error: No .jsonl files found in {data_dir} or {data_dir}/status/20**/")
        return

    print(f"Found {len(jsonl_files)} files to process.")
    all_events = []
    for file_path in sorted(jsonl_files):
        print(f"Processing file: {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_events.append(json.loads(line))

    # イベントをタイムスタンプでソートする
    all_events.sort(key=lambda x: x.get("created_at") or "")
    print(f"Loaded and sorted {len(all_events)} total events.")
    # ▲▲▲【ここまでが修正箇所】▲▲▲

    # --- 状態をシミュレートするための準備 ---
    try:
        with open(backlog_path, "r", encoding="utf-8") as f:
            all_tasks_list = json.load(f)
        all_tasks_db = {task["id"]: task for task in all_tasks_list}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading backlog file '{backlog_path}': {e}")
        return

    # --- メインループ ---
    trajectory = []
    for event in all_events:
        action_enum, action_details = map_event_to_action(event)

        # マッピングされたアクションがなければスキップ
        if not action_enum:
            continue

        event_timestamp = action_details.get("timestamp")
        developer = action_details.get("developer")

        if not all([event_timestamp, developer]):
            continue

        # --- イベント発生直前の「状態」を定義 ---
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

        # --- (状態, 行動)のペアを軌跡に追加 ---
        trajectory.append(
            {
                "state": current_state,
                "action": action_enum,
                "action_details": action_details,  # task_id, developer, timestampを含む
            }
        )

    # 一本の長い軌跡として保存
    trajectories = [trajectory] if trajectory else []

    with open(output_path, "wb") as f:
        pickle.dump(trajectories, f)
    print(
        f"Expert trajectories saved to {output_path}. Total trajectories: {len(trajectories)}, Total steps: {len(trajectory)}"
    )


if __name__ == "__main__":
    # 入力データディレクトリと、依存するファイルを指定
    INPUT_DATA_DIR = "./data"  # data/ 直下とstatus/20**/ 以下の両方を探索
    BACKLOG_FILE_PATH = "data/backlog.json"
    OUTPUT_TRAJECTORY_PATH = "data/expert_trajectories.pkl"

    main(INPUT_DATA_DIR, BACKLOG_FILE_PATH, OUTPUT_TRAJECTORY_PATH)
