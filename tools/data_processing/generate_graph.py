import json
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from torch_geometric.data import HeteroData


def debug_jsonl_format(file_path, max_lines=5):
    """JSONLファイルの形式をデバッグする"""
    print(f"\n--- デバッグ: {file_path} の最初の{max_lines}行 ---")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        print(f"Line {i+1}: keys={list(data.keys())}")
                        print(f"  Sample: {dict(list(data.items())[:3])}")
                    except json.JSONDecodeError as e:
                        print(f"Line {i+1}: JSON decode error: {e}")
    except Exception as e:
        print(f"File read error: {e}")


def generate_graph():
    """
    改良されたグラフデータ生成関数
    """
    data_dir = Path("data/status/")
    print(f"Data directory: {data_dir}")
    profile_path = Path("configs/dev_profiles.yaml")
    backlog_path = Path("data/backlog.json")
    graph_out = Path("data/graph.pt")

    # JSONLファイル検索
    all_jsonl_files = (
        [p for p in data_dir.glob("**/*.jsonl")] if data_dir.exists() else []
    )

    print("--- 取得対象のJSONLファイル一覧 ---")
    if all_jsonl_files:
        for file_path in all_jsonl_files[:5]:
            print(f"  - {file_path}")
        if len(all_jsonl_files) > 5:
            print(f"  ... and {len(all_jsonl_files) - 5} more files")

        # ▼▼▼【デバッグ追加】最初のファイルの形式を確認▼▼▼
        if all_jsonl_files:
            debug_jsonl_format(all_jsonl_files[0])
    else:
        print("  - JSONLファイルが見つかりません。バックログからデータを生成します。")

    print(f"--- 合計 {len(all_jsonl_files)} 個のファイル ---")

    # データを格納するための空のリストを準備
    prs = []
    issues = []
    unknown_events = []

    # ▼▼▼【修正箇所】JSONLファイル読み込みロジックを改良▼▼▼
    for file_path in all_jsonl_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                        # GitHub Archive形式のイベントデータを想定
                        event_type = data.get("type", "")

                        if event_type == "PullRequestEvent":
                            pr_data = data.get("payload", {}).get("pull_request", {})
                            if pr_data:
                                pr_obj = {
                                    "id": pr_data.get("id"),
                                    "author": data.get("actor", {}).get(
                                        "login", "unknown"
                                    ),
                                    "state": pr_data.get("state", "unknown"),
                                    "labels": [
                                        label.get("name", "")
                                        for label in pr_data.get("labels", [])
                                    ],
                                    "reviewers": [],  # GitHub Archiveには含まれていない場合が多い
                                    "kind": "pr",
                                }
                                prs.append(pr_obj)

                        elif event_type == "IssuesEvent":
                            issue_data = data.get("payload", {}).get("issue", {})
                            if issue_data:
                                issue_obj = {
                                    "id": issue_data.get("id"),
                                    "author": data.get("actor", {}).get(
                                        "login", "unknown"
                                    ),
                                    "state": issue_data.get("state", "unknown"),
                                    "labels": [
                                        label.get("name", "")
                                        for label in issue_data.get("labels", [])
                                    ],
                                    "assignee": (
                                        issue_data.get("assignee", {}).get("login")
                                        if issue_data.get("assignee")
                                        else None
                                    ),
                                    "kind": "issue",
                                }
                                issues.append(issue_obj)
                        else:
                            # その他のイベントタイプをカウント
                            unknown_events.append(event_type)

                    except json.JSONDecodeError as e:
                        if line_num < 5:  # 最初の5行のみエラーを表示
                            print(
                                f"JSON decode error in {file_path} line {line_num}: {e}"
                            )
                        continue

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    # ▼▼▼【デバッグ情報追加】▼▼▼
    print(
        f"\n処理完了: Total PRs loaded: {len(prs)}, Total Issues loaded: {len(issues)}"
    )
    if unknown_events:
        event_counts = {}
        for event in unknown_events:
            event_counts[event] = event_counts.get(event, 0) + 1
        print("Unknown event types found:")
        for event_type, count in sorted(
            event_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            print(f"  - {event_type}: {count}")
    # ▲▲▲【デバッグ情報ここまで】▲▲▲

    # プロフィール読み込み
    if not profile_path.exists():
        print(f"Error: Developer profiles not found: {profile_path}")
        return None

    with profile_path.open() as f:
        dev_profiles = yaml.safe_load(f)

    # バックログ読み込み
    if backlog_path.exists():
        with backlog_path.open() as f:
            backlog_tasks = json.load(f)
        print(f"Loaded {len(backlog_tasks)} tasks from backlog")
    else:
        print("Backlog file not found. Using empty task list.")
        backlog_tasks = []

    # === ノード構築 ===
    data = HeteroData()
    dev_set = set()
    task_ids = []
    task_features = []
    task_meta = []
    all_labels_list = [
        "bug",
        "enhancement",
        "documentation",
        "question",
        "performance",
        "CI/CD",
    ]
    label_to_idx = {label: i for i, label in enumerate(all_labels_list)}

    # PRノード構築
    for pr in prs:
        pr_id = f"pr_{pr.get('id', len(task_ids))}"
        task_ids.append(pr_id)

        author = pr.get("author", "unknown")
        dev_set.add(author)

        # PR特徴量: [1.0, 0.0] + [ラベル特徴量6次元] + [is_merged]
        labels = pr.get("labels", [])
        label_features = [0.0] * len(all_labels_list)
        for label in labels:
            if label in label_to_idx:
                label_features[label_to_idx[label]] = 1.0

        is_merged = 1.0 if pr.get("state") == "merged" else 0.0
        features = [1.0, 0.0] + label_features + [is_merged]

        task_features.append(features)
        task_meta.append((pr_id, author, "pr", pr))

    # Issueノード構築
    for issue in issues:
        issue_id = f"issue_{issue.get('id', len(task_ids))}"
        task_ids.append(issue_id)

        author = issue.get("author", "unknown")
        dev_set.add(author)

        # Issue特徴量: [0.0, 1.0] + [ラベル特徴量6次元] + [is_open]
        labels = issue.get("labels", [])
        label_features = [0.0] * len(all_labels_list)
        for label in labels:
            if label in label_to_idx:
                label_features[label_to_idx[label]] = 1.0

        is_open = 1.0 if issue.get("state") == "open" else 0.0
        features = [0.0, 1.0] + label_features + [is_open]

        task_features.append(features)
        task_meta.append((issue_id, author, "issue", issue))

    # ▼▼▼【修正箇所】開発者ノードが0個の場合のフォールバック▼▼▼
    if not dev_set:
        print(
            "⚠️  JSONLファイルから開発者が見つからないため、dev_profilesから開発者を追加します"
        )
        # dev_profilesから開発者を追加
        for dev_name in list(dev_profiles.keys())[:10]:  # 最初の10人
            dev_set.add(dev_name)

        # バックログタスクにダミーの作成者を設定
        for i, task in enumerate(
            backlog_tasks[: min(50, len(list(dev_profiles.keys())) * 3)]
        ):
            task_id = f"backlog_{task.get('id', len(task_ids))}"
            task_ids.append(task_id)

            # ランダムに開発者を割り当て
            import random

            author = random.choice(list(dev_set))

            features = [0.0, 1.0] + [0.0] * 6 + [1.0]  # Issue扱い、オープン状態
            task_features.append(features)
            task_meta.append((task_id, author, "backlog", task))
    # ▲▲▲【修正箇所ここまで】▲▲▲

    # 開発者ノード構築
    dev_names = sorted(list(dev_set))
    dev2idx = {name: i for i, name in enumerate(dev_names)}
    skill_features = []

    for name in dev_names:
        if name in dev_profiles:
            profile = dev_profiles[name]
            skill_data = profile.get("skill", {})
            skills = [
                float(skill_data.get("code", 0.5)),
                float(skill_data.get("review", 0.5)),
            ]

            langs = profile.get("lang_emb", [0.0, 0.0, 0.0])
            task_types = profile.get("task_types", [0.0, 0.0, 0.0])

            features = skills + langs + task_types
        else:
            features = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        skill_features.append(features)

    data["dev"].x = torch.tensor(skill_features, dtype=torch.float)
    data["dev"].node_id = dev_names

    # タスクノード
    data["task"].x = torch.tensor(task_features, dtype=torch.float)
    data["task"].node_id = task_ids

    print(f"Built nodes:")
    print(f"  - Developers: {len(dev_names)}")
    print(f"  - Tasks: {len(task_ids)}")

    # ▼▼▼【修正箇所】エッジ構築とバリデーションを改良▼▼▼
    # エッジ構築
    writes_edges = []
    reviews_edges = []
    assigned_edges = []

    for i, (task_id, author, kind, obj) in enumerate(task_meta):
        if author in dev2idx:
            dev_idx = dev2idx[author]

            # エッジの妥当性をチェック
            if dev_idx < len(dev_names) and i < len(task_ids):
                writes_edges.append([dev_idx, i])

            if kind == "pr" and "reviewers" in obj:
                for reviewer in obj["reviewers"]:
                    if reviewer in dev2idx:
                        reviewer_idx = dev2idx[reviewer]
                        if reviewer_idx < len(dev_names) and i < len(task_ids):
                            reviews_edges.append([reviewer_idx, i])

            if "assignee" in obj and obj["assignee"] in dev2idx:
                assignee_idx = dev2idx[obj["assignee"]]
                if assignee_idx < len(dev_names) and i < len(task_ids):
                    assigned_edges.append([assignee_idx, i])

    # エッジの妥当性を最終確認
    def validate_edges(edge_list, max_dev_idx, max_task_idx, edge_name):
        valid_edges = []
        for dev_idx, task_idx in edge_list:
            if 0 <= dev_idx < max_dev_idx and 0 <= task_idx < max_task_idx:
                valid_edges.append([dev_idx, task_idx])
            else:
                print(
                    f"Warning: Invalid edge in {edge_name}: dev={dev_idx} (max={max_dev_idx-1}), task={task_idx} (max={max_task_idx-1})"
                )
        return valid_edges

    writes_edges = validate_edges(writes_edges, len(dev_names), len(task_ids), "writes")
    reviews_edges = validate_edges(
        reviews_edges, len(dev_names), len(task_ids), "reviews"
    )
    assigned_edges = validate_edges(
        assigned_edges, len(dev_names), len(task_ids), "assigned"
    )

    # エッジテンソル作成
    def to_edge_index(edge_list):
        if not edge_list:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    data[("dev", "writes", "task")].edge_index = to_edge_index(writes_edges)
    data[("dev", "reviews", "task")].edge_index = to_edge_index(reviews_edges)
    data[("dev", "assigned", "task")].edge_index = to_edge_index(assigned_edges)

    # 逆方向エッジを正しく作成
    if writes_edges:
        # [dev_idx, task_idx] を [task_idx, dev_idx] に変換
        data[("task", "written_by", "dev")].edge_index = data[
            ("dev", "writes", "task")
        ].edge_index.flip([0])
    else:
        data[("task", "written_by", "dev")].edge_index = torch.empty(
            (2, 0), dtype=torch.long
        )

    # ▼▼▼【デバッグ情報追加】エッジの妥当性を確認▼▼▼
    print(f"Built edges:")
    print(f"  - Writes: {len(writes_edges)}")
    print(f"  - Reviews: {len(reviews_edges)}")
    print(f"  - Assigned: {len(assigned_edges)}")

    # エッジの範囲を確認
    if writes_edges:
        dev_writes_edge = data[("dev", "writes", "task")].edge_index
        task_written_edge = data[("task", "written_by", "dev")].edge_index

        print(f"Edge validation:")
        print(
            f"  - dev->task edges: dev range [0, {dev_writes_edge[0].max().item()}], task range [0, {dev_writes_edge[1].max().item()}]"
        )
        print(
            f"  - task->dev edges: task range [0, {task_written_edge[0].max().item()}], dev range [0, {task_written_edge[1].max().item()}]"
        )
        print(
            f"  - Expected: dev nodes [0, {len(dev_names)-1}], task nodes [0, {len(task_ids)-1}]"
        )
    # ▲▲▲【デバッグ情報ここまで】▲▲▲
    # ▲▲▲【修正箇所ここまで】▲▲▲

    # データ保存
    graph_out.parent.mkdir(exist_ok=True)
    torch.save(data, graph_out)
    print(f"✅ グラフデータを保存しました: {graph_out}")

    return data


def main():
    """メイン関数"""
    try:
        data = generate_graph()
        if data is not None:
            print("\n=== 生成されたグラフデータの概要 ===")
            print(data)
            for node_type, x in data.x_dict.items():
                print(f"Node type '{node_type}': {x.shape}")
            print("✅ グラフデータ生成完了")
        else:
            print("❌ グラフデータ生成に失敗しました")
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
