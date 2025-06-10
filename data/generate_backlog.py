import datetime
from kazoo.envs.task import Task
from pathlib import Path
import json

def load_tasks(path):
    """github_data.json から OPENなPRを抽出し、Taskインスタンスのリストを返す"""
    path = Path(path)

    with path.open() as f:
        raw = json.load(f)
    prs = raw.get("prs", [])

    backlog = []
    for pr in prs:
        if pr.get("state") != "OPEN":
            continue

        created_at = datetime.datetime.strptime(pr.get("createdAt"), "%Y-%m-%dT%H:%M:%SZ")

        task = Task(
            id=pr["number"],
            title=pr.get("title", ""),
            author=pr.get("author", {}).get("login"),
            complexity=min(len(pr.get("body", "")) // 100 + 1, 5),
            created_at=created_at,
            labels=[l["name"] for l in pr.get("labels", {}).get("nodes", [])]
        )

        backlog.append(task)

    return backlog
