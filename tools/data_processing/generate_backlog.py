import glob
import json
import os
from datetime import datetime


def parse_datetime(timestamp_str: str | None) -> datetime | None:
    """ISO 8601形式のタイムスタンプ文字列をdatetimeオブジェクトに変換する。"""
    if not timestamp_str:
        return None
    try:
        if timestamp_str.endswith("Z"):
            timestamp_str = timestamp_str[:-1] + "+00:00"
        return datetime.fromisoformat(timestamp_str)
    except (ValueError, TypeError):
        return None


def generate_full_backlog(data_dir, output_path):
    """
    全てのイベントログを処理し、各タスクの完全なライフサイクル情報を含む
    包括的なバックログファイルを作成する。

    従来のbackward compatibility関数
    """
    return process_multiple_directories(data_dir, output_path)


def process_multiple_directories(base_data_dir, output_path):
    """
    複数のディレクトリから全ての.jsonlファイルを処理する
    """
    print(f"Starting to generate full backlog from base directory: {base_data_dir}")

    # 全てのタスク情報をIDをキーにして保持する辞書
    tasks_db = {}

    # データディレクトリ内の全ての.jsonlファイルを検索
    # ルートディレクトリ（data/）の.jsonlファイル
    jsonl_files = glob.glob(os.path.join(base_data_dir, "*.jsonl"))

    # status/サブディレクトリの.jsonlファイル
    status_dir = os.path.join(base_data_dir, "status")
    if os.path.exists(status_dir):
        for year_dir in os.listdir(status_dir):
            year_path = os.path.join(status_dir, year_dir)
            if os.path.isdir(year_path):
                year_files = glob.glob(os.path.join(year_path, "*.jsonl"))
                jsonl_files.extend(year_files)

    if not jsonl_files:
        print(f"Error: No .jsonl files found in directory: {base_data_dir}")
        return

    print(f"Found {len(jsonl_files)} files to process.")

    for file_path in sorted(jsonl_files):
        print(f"Processing file: {file_path}...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # IssueまたはPRに関するイベントから情報を抽出
                    issue_data = None
                    event_type = event.get("type")
                    if event_type in [
                        "IssuesEvent",
                        "IssueCommentEvent",
                    ] and "issue" in event.get("payload", {}):
                        issue_data = event["payload"]["issue"]
                    elif event_type in [
                        "PullRequestEvent",
                        "PullRequestReviewCommentEvent",
                    ] and "pull_request" in event.get("payload", {}):
                        issue_data = event["payload"]["pull_request"]

                    if issue_data and "id" in issue_data:
                        task_id = issue_data["id"]

                        # データベースにタスクがなければ新規作成
                        if task_id not in tasks_db:
                            tasks_db[task_id] = issue_data
                        else:
                            # 既存のタスク情報を最新の状態に更新
                            tasks_db[task_id].update(issue_data)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # 最終的なバックログリストを作成
    final_backlog = list(tasks_db.values())
    print(f"Generated a backlog with {len(final_backlog)} unique tasks.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_backlog, f, indent=2)

    print(f"✅ Successfully generated full backlog at: {output_path}")


if __name__ == "__main__":
    import sys

    # コマンドライン引数がある場合はそれを使用、ない場合はデフォルト
    if len(sys.argv) > 1:
        INPUT_DATA_DIR = sys.argv[1]
    else:
        INPUT_DATA_DIR = "./data/"

    if len(sys.argv) > 2:
        OUTPUT_JSON_PATH = sys.argv[2]
    else:
        OUTPUT_JSON_PATH = "data/backlog.json"

    process_multiple_directories(INPUT_DATA_DIR, OUTPUT_JSON_PATH)
