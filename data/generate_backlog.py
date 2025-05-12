import json
import re
from pathlib import Path

# 入出力ファイルパス
root = Path(__file__).resolve().parents[1]
input_path = root / "data/github_data.json"
output_path = root / "data/backlog.json"

# 読み込み
with open(input_path) as f:
    data = json.load(f)

prs = data["data"]["repository"]["pullRequests"]["nodes"]
backlog = []

# タグのキーワードマッピング
tag_keywords = {
    "fix|bug|error": "bugfix",
    "refactor|clean": "refactor",
    "doc|readme|comment": "docs",
    "test|assert": "test",
    "feat|feature|add": "feature"
}

def extract_tags(title):
    tags = []
    for key, tag in tag_keywords.items():
        if re.search(rf"\b{key}\b", title, re.IGNORECASE):
            tags.append(tag)
    return tags or ["misc"]

# PRごとに backlog エントリを作成
for pr in prs:
    backlog.append({
        "id": pr["number"],
        "title": pr["title"],
        "complexity": len(pr["title"]) % 3 + 1,  # タイトル長をベースに1〜3
        "tags": extract_tags(pr["title"]),
        "author": pr["author"]["login"]
    })

# 保存
with open(output_path, "w") as f:
    json.dump(backlog, f, indent=2)

print(f"✅ backlog.json を生成しました → {output_path}")