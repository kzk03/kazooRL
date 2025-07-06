import glob
import json
import os
from collections import defaultdict

import yaml


def generate_developer_profiles(data_dir, output_path, exclude_years=None):
    """
    指定されたディレクトリにある全ての .jsonl ファイルを読み込み、
    開発者ごとの静的な特徴量を事前計算して、dev_profiles.yamlとして出力する。

    Args:
        data_dir: データディレクトリのパス
        output_path: 出力ファイルのパス
        exclude_years: 除外する年のリスト（例: ["2022"]）

    計算する特徴量:
    - label_affinity: どのラベルのタスクをどれくらいの割合で完了させたか
    - touched_files: これまでに編集したことのあるファイルの一覧
    - total_merged_prs: マージされたPRの総数
    - total_lines_changed: マージされたPRでの総変更行数（追加行数+削除行数）
    - collaboration_network_size: 一緒にPRで作業した開発者の数（社会的つながり）
    - comment_interactions: 他の開発者のIssue/PRにコメントした回数（社会的参加）
    - cross_issue_activity: 複数のIssueにまたがる活動度（コミュニティ参加）
    """
    if exclude_years is None:
        exclude_years = []

    print(f"Starting to generate developer profiles from directory: {data_dir}")
    if exclude_years:
        print(f"Excluding years: {exclude_years}")

    # data_dir内の全ての .jsonl ファイルを取得（status/20**以下も含む）
    # 例: "data/status/2022/gharchive_docker_compose_events_2022-01.jsonl"
    jsonl_files = []

    # data_dir直下のjsonlファイル
    direct_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    jsonl_files.extend(direct_files)

    # data_dir/status/20** 以下のjsonlファイル
    status_dir = os.path.join(data_dir, "status")
    if os.path.exists(status_dir):
        for year_dir in os.listdir(status_dir):
            # 除外する年をスキップ
            if year_dir in exclude_years:
                print(f"Skipping year directory: {year_dir}")
                continue

            year_path = os.path.join(status_dir, year_dir)
            if os.path.isdir(year_path):
                year_files = glob.glob(os.path.join(year_path, "*.jsonl"))
                jsonl_files.extend(year_files)

    if not jsonl_files:
        print(
            f"Error: No .jsonl files found in directory: {data_dir} or {data_dir}/status/20**/"
        )
        return

    print(f"Found {len(jsonl_files)} files to process.")

    dev_stats = defaultdict(
        lambda: {
            "label_counts": defaultdict(int),
            "touched_files": set(),
            "merged_pr_count": 0,
            "total_lines_changed": 0,
            # 社会的つながりの特徴量
            "collaborators": set(),  # 一緒に作業した開発者
            "comment_interactions": 0,  # コメント数
            "issues_participated": set(),  # 参加したIssue/PR
        }
    )

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
                        # 壊れた行はスキップ
                        continue

                    # PullRequestEvent処理
                    if (
                        event.get("type") == "PullRequestEvent"
                        and event.get("payload", {}).get("action") == "closed"
                        and event.get("payload", {})
                        .get("pull_request", {})
                        .get("merged")
                    ):

                        pr = event["payload"]["pull_request"]
                        developer = pr.get("user", {}).get("login")
                        if not developer:
                            continue

                        dev_stats[developer]["merged_pr_count"] += 1
                        dev_stats[developer]["issues_participated"].add(
                            pr.get("number", 0)
                        )

                        # 行数変更を追加
                        additions = pr.get("additions", 0)
                        deletions = pr.get("deletions", 0)
                        dev_stats[developer]["total_lines_changed"] += (
                            additions + deletions
                        )

                        for label in pr.get("labels", []):
                            if "name" in label:
                                dev_stats[developer]["label_counts"][label["name"]] += 1

                        # ファイルリストの取得 (get_github_data.pyでの取得が前提)
                        for file_info in pr.get("files", []):
                            if "filename" in file_info:
                                dev_stats[developer]["touched_files"].add(
                                    file_info["filename"]
                                )

                        # 協力者情報を収集（assigneesやreviewersから）
                        for assignee in pr.get("assignees", []):
                            assignee_login = assignee.get("login")
                            if assignee_login and assignee_login != developer:
                                dev_stats[developer]["collaborators"].add(
                                    assignee_login
                                )
                                dev_stats[assignee_login]["collaborators"].add(
                                    developer
                                )

                        for reviewer in pr.get("requested_reviewers", []):
                            reviewer_login = reviewer.get("login")
                            if reviewer_login and reviewer_login != developer:
                                dev_stats[developer]["collaborators"].add(
                                    reviewer_login
                                )
                                dev_stats[reviewer_login]["collaborators"].add(
                                    developer
                                )

                    # IssueCommentEvent処理（社会的相互作用）
                    elif event.get("type") == "IssueCommentEvent":
                        actor_login = event.get("actor", {}).get("login")
                        issue_number = (
                            event.get("payload", {}).get("issue", {}).get("number")
                        )
                        issue_author = (
                            event.get("payload", {})
                            .get("issue", {})
                            .get("user", {})
                            .get("login")
                        )

                        if actor_login and issue_number:
                            dev_stats[actor_login]["comment_interactions"] += 1
                            dev_stats[actor_login]["issues_participated"].add(
                                issue_number
                            )

                            # 異なる開発者のIssue/PRにコメントした場合、協力者として記録
                            if issue_author and issue_author != actor_login:
                                dev_stats[actor_login]["collaborators"].add(
                                    issue_author
                                )
                                dev_stats[issue_author]["collaborators"].add(
                                    actor_login
                                )
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    final_profiles = {}
    for dev, stats in dev_stats.items():
        total_labels = sum(stats["label_counts"].values())
        label_affinity = (
            {
                label: count / total_labels
                for label, count in stats["label_counts"].items()
            }
            if total_labels > 0
            else {}
        )

        final_profiles[dev] = {
            "skills": ["python"],
            "label_affinity": label_affinity,
            "touched_files": sorted(list(stats["touched_files"])),
            "total_merged_prs": stats["merged_pr_count"],
            "total_lines_changed": stats["total_lines_changed"],
            # 社会的つながりの特徴量
            "collaboration_network_size": len(stats["collaborators"]),
            "collaborators": sorted(
                list(stats["collaborators"])
            ),  # 協力者の具体的なリスト
            "comment_interactions": stats["comment_interactions"],
            "cross_issue_activity": len(stats["issues_participated"]),
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(final_profiles, f, default_flow_style=False, allow_unicode=True)

    print(
        f"✅ Successfully generated developer profiles for {len(final_profiles)} developers at: {output_path}"
    )


if __name__ == "__main__":
    # データ取得スクリプトが出力したディレクトリを指定
    INPUT_DATA_DIR = "./data"

    # 2022年のデータを除外してトレーニング用プロファイルを生成
    exclude_years = ["2022"]

    # トレーニング用プロファイル（2022年除外）
    training_output = "./configs/dev_profiles_training.yaml"
    print(f"Generating training profiles (excluding {exclude_years})...")
    generate_developer_profiles(INPUT_DATA_DIR, training_output, exclude_years)

    # テスト用プロファイル（2022年のみ）
    test_output = "./configs/dev_profiles_test_2022.yaml"
    print(f"\nGenerating test profiles (2022 only)...")
    generate_developer_profiles(
        INPUT_DATA_DIR,
        test_output,
        exclude_years=["2019", "2020", "2021", "2023", "2024"],
    )

    # 従来の完全なプロファイルも生成（後方互換性のため）
    OUTPUT_YAML_PATH = "./configs/dev_profiles.yaml"
    print(f"\nGenerating complete profiles (all years)...")
    generate_developer_profiles(INPUT_DATA_DIR, OUTPUT_YAML_PATH)
