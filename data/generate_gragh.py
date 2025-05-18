import json
import torch
from pathlib import Path
from torch_geometric.data import HeteroData

# === ファイル読み込み ===
root = Path(__file__).resolve().parents[1]
data_path = root / "data/github_data.json"
output_path = root / "data/graph.pt"

with open(data_path) as f:
    raw = json.load(f)

prs = raw["data"]["repository"]["pullRequests"]["nodes"]

# === ノード・エッジ抽出 ===
dev_set = set()
task_set = set()
edges_authors = []
edges_reviewers = []

for pr in prs:
    task_id = f"task_{pr['number']}"
    task_set.add(task_id)

    author = pr["author"]["login"]
    dev_set.add(author)
    edges_authors.append((author, task_id))

    for review in pr.get("reviews", {}).get("nodes", []):
        reviewer = review["author"]["login"]
        dev_set.add(reviewer)
        edges_reviewers.append((reviewer, task_id))

# === ノードIDを数値IDに変換 ===
dev_list = sorted(dev_set)
task_list = sorted(task_set)
dev2idx = {dev: i for i, dev in enumerate(dev_list)}
task2idx = {task: i for i, task in enumerate(task_list)}

# === PyGのグラフ構造 ===
data = HeteroData()

# ダミーのノード特徴（必要に応じて dev_profiles.yaml を使って置き換えてもOK）
data["dev"].x = torch.ones((len(dev_list), 5))
data["task"].x = torch.ones((len(task_list), 5))

# エッジ（author: dev → task）
data["dev", "writes", "task"].edge_index = torch.tensor([
    [dev2idx[d] for d, t in edges_authors],
    [task2idx[t] for d, t in edges_authors]
], dtype=torch.long)

# エッジ（review: dev → task）
data["dev", "reviews", "task"].edge_index = torch.tensor([
    [dev2idx[d] for d, t in edges_reviewers],
    [task2idx[t] for d, t in edges_reviewers]
], dtype=torch.long)

# 保存
torch.save(data, output_path)
print(f"✅ グラフデータを保存しました → {output_path}")
