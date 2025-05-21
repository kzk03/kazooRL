import json
from pathlib import Path

# === 設定 ===
data_path = Path("data/github_data.json")  # 必要に応じて変更
out_path = Path("data/backlog.json")

# === データ読み込み ===
with data_path.open() as f:
    raw = json.load(f)
prs = raw.get("prs", [])

# === OPENなPRのみ抽出 ===
backlog = []
for pr in prs:
    if pr.get("state") != "OPEN":
        continue

    backlog.append({
        "id": pr["number"],
        "title": pr.get("title", ""),
        "labels": [l["name"] for l in pr.get("labels", {}).get("nodes", [])],
        "author": pr.get("author", {}).get("login"),
        "complexity": min(len(pr.get("body", "")) // 100 + 1, 5),  # 簡易スコア
        "createdAt": pr.get("createdAt")
    })

# === 保存 ===
out_path.parent.mkdir(exist_ok=True)
with out_path.open("w") as f:
    json.dump(backlog, f, indent=2)

print(f"✅ backlog.json を保存しました → {out_path}（件数: {len(backlog)}）")
