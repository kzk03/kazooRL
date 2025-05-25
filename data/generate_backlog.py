import json
from pathlib import Path


def load_tasks(path):
    """github_data.json から OPENなPRを抽出し、整形されたタスクリストを返す"""
    path = Path(path)

    with path.open() as f:
        raw = json.load(f)
    prs = raw.get("prs", [])

    backlog = []
    for pr in prs:
        if pr.get("state") != "OPEN":
            continue

        backlog.append({
            "id": pr["number"],
            "title": pr.get("title", ""),
            "labels": [l["name"] for l in pr.get("labels", {}).get("nodes", [])],
            "author": pr.get("author", {}).get("login"),
            "complexity": min(len(pr.get("body", "")) // 100 + 1, 5),
            "createdAt": pr.get("createdAt")
        })

    return backlog
