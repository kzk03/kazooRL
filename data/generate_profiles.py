import json
from collections import defaultdict
from pathlib import Path

import yaml

# パス設定
root = Path(__file__).resolve().parents[1]
data_path = root / "data/github_data.json"
output_path = root / "configs/dev_profiles.yaml"

# JSON読み込み
with open(data_path) as f:
    data = json.load(f)

# スキル初期化
profiles = defaultdict(lambda: {
    "skill": {"code": 0.0, "review": 0.0},
    "lang_emb": [1.0, 0.0, 0.0],
    "task_types": [10, 3, 2]
})

# PRごとに処理
prs = data["data"]["repository"]["pullRequests"]["nodes"]
for pr in prs:
    author = pr["author"]["login"]
    profiles[author]["skill"]["code"] += 0.01

    for review in pr.get("reviews", {}).get("nodes", []):
        reviewer = review["author"]["login"]
        profiles[reviewer]["skill"]["review"] += 0.01

# スキルの最大値を1.0に制限
for dev in profiles:
    for key in ["code", "review"]:
        profiles[dev]["skill"][key] = round(min(profiles[dev]["skill"][key], 1.0), 2)

# 書き出し
with open(output_path, "w") as f:
    yaml.dump(dict(profiles), f, sort_keys=False)

print(f"✅ dev_profiles.yaml を生成しました → {output_path}")
