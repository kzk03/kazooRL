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

# ダミー特徴（必要なら dev_profiles.yaml 等から生成可能）
data["dev"].x = torch.ones((len(dev_list), 5))
data["task"].x = torch.ones((len(task_list), 5))

# === エッジ定義（dev → task）
author_edges = torch.tensor([
    [dev2idx[d] for d, t in edges_authors],
    [task2idx[t] for d, t in edges_authors]
], dtype=torch.long)

review_edges = torch.tensor([
    [dev2idx[d] for d, t in edges_reviewers],
    [task2idx[t] for d, t in edges_reviewers]
], dtype=torch.long)

data["dev", "writes", "task"].edge_index = author_edges
data["dev", "reviews", "task"].edge_index = review_edges

# === エッジ定義（task → dev）← 双方向追加！！
data["task", "written_by", "dev"].edge_index = author_edges.flip(0)
data["task", "reviewed_by", "dev"].edge_index = review_edges.flip(0)

# 保存
torch.save(data, output_path)
print(f"✅ グラフを保存しました → {output_path}")
print(f"🧠 dev数: {len(dev_list)}, task数: {len(task_list)}")