import json
import random
from pathlib import Path

import yaml

with open("data/github_data.json") as f:
    prs = json.load(f)["data"]["repository"]["pullRequests"]["nodes"]

devs = set()
for pr in prs:
    devs.add(pr["author"]["login"])
    for r in pr.get("reviews", {}).get("nodes", []):
        if r.get("author"):
            devs.add(r["author"]["login"])

profiles = {}
for dev in sorted(devs):
    profiles[dev] = {
        "skill": {"code": round(random.uniform(0.3, 1.0), 2), "review": round(random.uniform(0.3, 1.0), 2)},
        "lang_emb": [1.0, 0.0, 0.0],
        "task_types": [random.randint(1, 5) for _ in range(3)],
    }

with open("configs/dev_profiles.yaml", "w") as f:
    yaml.dump(profiles, f)

print(f"✅ {len(profiles)} 件の開発者プロファイルを生成しました。→ configs/dev_profiles.yaml")
