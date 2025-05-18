import json
import torch

# グラフとPRデータ
data = torch.load("data/graph.pt", weights_only=False)
dev_names = set(data["dev"].node_id)

with open("data/github_data.json") as f:
    prs = json.load(f)["data"]["repository"]["pullRequests"]["nodes"]

reviewers = set()
for pr in prs:
    for r in pr.get("reviews", {}).get("nodes", []):
        if r.get("author"):
            reviewers.add(r["author"]["login"])

overlap = dev_names & reviewers
print(f"✅ node_idに含まれるレビュアー数: {len(overlap)} / {len(reviewers)}")
print("🔁 一致レビュアー:", list(overlap))
