import json
from pathlib import Path

import torch
import yaml
from torch_geometric.data import HeteroData

# === パス設定
root = Path(__file__).resolve().parents[1]
graph_out = root / "data/graph.pt"
github_path = root / "data/github_data.json"
profile_path = root / "configs/dev_profiles.yaml"
backlog_path = root / "data/backlog.json"

# === データ読み込み
with open(github_path) as f:
    pr_data = json.load(f)["data"]["repository"]["pullRequests"]["nodes"]
with open(profile_path) as f:
    profiles = yaml.safe_load(f)
with open(backlog_path) as f:
    backlog = json.load(f)

# dev_list を GitHubデータから自動抽出
dev_list = sorted(set(
    pr["author"]["login"] for pr in pr_data
    if pr.get("author")
).union(*[
    {r["author"]["login"] for r in pr.get("reviews", {}).get("nodes", []) if r.get("author")}
    for pr in pr_data
]))

task_list = sorted([f"task_{pr['number']}" for pr in pr_data])
dev2idx = {dev: i for i, dev in enumerate(dev_list)}
task2idx = {task: i for i, task in enumerate(task_list)}

# === エッジ構築
edges_author = []
edges_review = []

for pr in pr_data:
    task_id = f"task_{pr['number']}"
    author = pr["author"]["login"]
    if author in dev2idx:
        edges_author.append((author, task_id))
    for review in pr.get("reviews", {}).get("nodes", []):
        reviewer = review["author"]["login"]
        if reviewer in dev2idx:
            edges_review.append((reviewer, task_id))

# === devノード特徴量
dev_feats = []
for dev in dev_list:
    p = profiles[dev]
    feat = [
        p["skill"]["code"],
        p["skill"]["review"],
        *p["lang_emb"],
        *p["task_types"],
    ]
    dev_feats.append(feat)

# === taskノード特徴量（+6次元パディングで8次元に統一）
tag_map = {"bugfix": 0, "feature": 1, "refactor": 2, "docs": 3, "test": 4, "misc": 5}
import hashlib

# 疑似的に complexity & tag を生成（ランダム性ありだが再現性も保つ）
tag_map = {"bug": 0, "feature": 1, "refactor": 2, "docs": 3, "test": 4, "misc": 5}
task_feats = []

for pr in pr_data:
    title = pr["title"].lower()
    pr_id = pr["number"]

    # 疑似 complexity（1〜3）
    complexity = (hash(title) % 3) + 1

    # 疑似 tag 判定
    tag_idx = 5  # default to 'misc'
    for keyword, idx in tag_map.items():
        if keyword in title:
            tag_idx = idx
            break

    feat = [complexity, tag_idx] + [0.0] * 6
    task_feats.append(feat)
# === グラフ構築
data = HeteroData()
data["dev"].x = torch.tensor(dev_feats, dtype=torch.float)
data["dev"].node_id = dev_list  # 🔥 GitHub login名を保存
data["task"].x = torch.tensor(task_feats, dtype=torch.float)

# === エッジ定義（dev → task）
src_a = [dev2idx[d] for d, t in edges_author]
dst_a = [task2idx[t] for d, t in edges_author]
data["dev", "writes", "task"].edge_index = torch.tensor([src_a, dst_a], dtype=torch.long)

src_r = [dev2idx[d] for d, t in edges_review]
dst_r = [task2idx[t] for d, t in edges_review]
data["dev", "reviews", "task"].edge_index = torch.tensor([src_r, dst_r], dtype=torch.long)

# === 双方向エッジ（task → dev）
data["task", "written_by", "dev"].edge_index = torch.tensor([dst_a, src_a], dtype=torch.long)
data["task", "reviewed_by", "dev"].edge_index = torch.tensor([dst_r, src_r], dtype=torch.long)

# === 保存
torch.save(data, graph_out)
print(f"✅ graph.pt を保存しました → {graph_out}")
print(f"👤 devノード数: {len(dev_list)}, 🧩 taskノード数: {len(task_list)}")


# import networkx as nx

# G = nx.DiGraph()
# for node_type in data.node_types:
#     for i in range(data[node_type].num_nodes):
#         G.add_node(f"{node_type}_{i}", node_type=node_type)

# for edge_type in data.edge_types:
#     src, rel, dst = edge_type
#     edge_index = data[edge_type].edge_index
#     for i in range(edge_index.size(1)):
#         src_node = f"{src}_{edge_index[0, i].item()}"
#         dst_node = f"{dst}_{edge_index[1, i].item()}"
#         G.add_edge(src_node, dst_node)

# isolated_tasks = [n for n in G.nodes if n.startswith("task_") and G.degree[n] == 0]
# print(f"🟡 孤立した task ノード数: {len(isolated_tasks)}")
# if isolated_tasks:
#     print("🔍 例:", isolated_tasks[:5])